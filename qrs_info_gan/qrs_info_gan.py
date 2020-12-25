import datetime
from torch import nn
import data_loader
import drawer
import params
import torch
from torch.autograd import Variable
import torch.utils
import itertools
import numpy as np
import dataset_creator

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def saveModel(model, name, path):
        torch.save(model.state_dict(), f"{path}/{name}.pt")


def loadModel(model, name, path):
        model.load_state_dict(torch.load(f"{path}/{name}.pt"))
        model.eval()
        return model


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    return y_cat


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, patch_len):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.code_dim = code_dim

        input_dim = latent_dim + n_classes + code_dim

        self.init_len = patch_len // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_len))

        self.conv_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 1, 3, stride=1, padding=1),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_len)

        out = self.conv_block(out)
        return out

    def sample_input_numpy(self, batch_size):
        z = np.random.normal(0, 1, (batch_size, self.latent_dim))

        label_input_int = np.random.randint(0, self.n_classes, batch_size)
        label_input_one_hot = to_categorical(label_input_int, num_columns=self.n_classes)

        code_input = np.random.uniform(-1, 1, (batch_size, self.code_dim))
        return z, label_input_int, label_input_one_hot, code_input


    def generate_ecgs(self, batch_size):
        z, _, label_input_one_hot, code_input = self.sample_input_numpy(batch_size)
        label_input_one_hot = Variable(torch.FloatTensor(label_input_one_hot))
        code_input = Variable(torch.FloatTensor(code_input))
        z = Variable(torch.FloatTensor(z))
        gen_ecgs = self.forward(z, label_input_one_hot, code_input)
        return gen_ecgs


class Discriminator(nn.Module):
    def __init__(self, n_classes, code_dim, patch_len):
        super(Discriminator, self).__init__()

        def downscale_block(in_filters, out_filters, bn=False):
            block = [nn.Conv1d(in_filters, out_filters, 9, 2, 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *downscale_block(1, 16, bn=False),
            *downscale_block(16, 32),
            *downscale_block(32, 64),
            *downscale_block(64, 128))

        # The lenght of downsampled ecg patch
        ds_len = patch_len // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_len, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_len, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_len, code_dim))


    def forward(self, ecg):
        out = self.model(ecg)
        out = out.view(out.shape[0], -1)

        self.latentRes = out

        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


    def getLatentRes(self):
        m = nn.Flatten()
        result = m(self.latentRes)
        result = result.detach().numpy()
        return result


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(pr, dataset):
    torch.manual_seed(111)

    best_loss = None
    cuda = True if torch.cuda.is_available() else False

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Initialize generator and discriminator
    generator = Generator(latent_dim=pr.latent_dim,
                          n_classes=pr.n_classes, code_dim=pr.code_dim,
                          patch_len=pr.patch_len)
    discriminator = Discriminator(n_classes=pr.n_classes,
                                  code_dim=pr.code_dim,
                                  patch_len=pr.patch_len)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=pr.lr, betas=(pr.b1, pr.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=pr.lr, betas=(pr.b1, pr.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=pr.lr, betas=(pr.b1, pr.b2)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=pr.batch_size, shuffle=True)

    disLoss = []
    genLoss = []
    infoLoss = []

    for epoch in range(pr.num_epochs):
        for i, (ecgs) in enumerate(dataloader):

            # Adversarial ground truths
            batch_size = ecgs.shape[0]
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z, _, label_input_one_hot, code_input = generator.sample_input_numpy(batch_size)

            label_input_one_hot = Variable(FloatTensor(label_input_one_hot))
            code_input = Variable(FloatTensor(code_input))
            #z = torch.from_numpy(z)
            z = Variable(FloatTensor(z))

            # Generate a batch of images
            gen_ecgs = generator(z, label_input_one_hot, code_input)
            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_ecgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Loss for real images
            real_pred, _, _ = discriminator(ecgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_ecgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()
            # Sample labels

            z, label_input_int, label_input_one_hot, code_input = generator.sample_input_numpy(batch_size)
            z = Variable(FloatTensor(z))
            code_input = Variable(FloatTensor(code_input))
            gt_labels = Variable(LongTensor(label_input_int), requires_grad=False)
            label_input_one_hot = Variable(FloatTensor(label_input_one_hot))

            gen_ecgs = generator(z, label_input_one_hot, code_input)
            _, pred_label, pred_code = discriminator(gen_ecgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            print(f"Epoch: {epoch} Loss D.: {d_loss.item()}")
            print(f"Epoch: {epoch} Loss G.: {g_loss.item()}")
            print(f"Epoch: {epoch} Loss I.: {info_loss.item()}")
            print(f"Iteration: {i}")

            if i == dataloader.__len__() - 1:
                genLoss.append(g_loss.item())
                disLoss.append(d_loss.item())
                infoLoss.append(info_loss.item())

    print("success")
    return discriminator, generator, disLoss, genLoss, infoLoss


def experiment(pr):
    now = datetime.datetime.now()
    path = drawer.makeDir(now)

    with open(path + "/params.txt", "w") as text_file:
        text_file.write(str(pr._asdict()))

    dataset_object = data_loader.QrsDataLoader(pr.patch_len//2)
    discriminator, generator, disLoss, genLoss, infoLoss = train(pr, dataset_object)

    saveModel(generator, "generator", path)
    saveModel(discriminator, "discriminator", path)

    errDrr = drawer.ErrDrawer(path, 1)

    errDrr.add(disLoss, "red", "dis")
    errDrr.add(genLoss, "blue", "gen")
    errDrr.add(infoLoss, "green", "inf")

    errDrr.save()

    drr = drawer.SignalDrawer(4, path, 0)
    realqrs = data_loader.completeSignal(dataset_object.getTestCenter())
    drr.add(realqrs)

    a = torch.zeros([1, 10])
    b = torch.zeros([1, 5])
    c = torch.zeros([1, 2])

    output = generator(a, b, c)
    output.squeeze_(0)
    output.squeeze_(0)
    output = output.detach().numpy()

    print(np.shape(output))
    imqrs = data_loader.completeSignal(output)
    print(np.shape(imqrs))

    drr.add(imqrs)

    testSignal = dataset_object.getTestSignal()
    drr.add(testSignal)

    result = []
    for i in range(pr.patch_len//2, 5000 - pr.patch_len//2 + 1):
        signal = data_loader.cutSignal(testSignal, i, pr.patch_len//2)
        signal = torch.from_numpy(signal)
        signal = signal.unsqueeze(0)
        signal = signal.unsqueeze(0)
        tempResult, _, _ = discriminator(signal)
        discriminator.getLatentRes()
        tempResult = tempResult.detach().numpy()
        result.append(tempResult[0])

    result = np.resize(result, (len(result),))
    result = data_loader.completeSignal(result)

    drr.add(result)
    drr.save()
    drr.clear()


    batch_size = 1500
    gen_ecgs = generator.generate_ecgs(batch_size)
    generator = Generator(latent_dim=pr.latent_dim,
                          n_classes=pr.n_classes, code_dim=pr.code_dim,
                          patch_len=pr.patch_len)
    generator.apply(weights_init_normal)
    gen_not_ecgs = generator.generate_ecgs(batch_size)


    discriminator(gen_ecgs)
    results = discriminator.getLatentRes()
    results = results.flatten()
    num_bins = 500
    n, bins, patches = plt.hist(results, num_bins, facecolor='blue')
    plt.show()
    plt.gcf().clear()


    discriminator(gen_not_ecgs)
    results = discriminator.getLatentRes()
    results = results.flatten()
    num_bins = 500
    n, bins, patches = plt.hist(results, num_bins, facecolor='blue')
    plt.show()
    plt.gcf().clear()

    discriminator = Discriminator(n_classes=pr.n_classes,
                                  code_dim=pr.code_dim,
                                  patch_len=pr.patch_len)
    discriminator.apply(weights_init_normal)

    discriminator(gen_ecgs)
    results = discriminator.getLatentRes()
    results = results.flatten()
    num_bins = 500
    n, bins, patches = plt.hist(results, num_bins, facecolor='blue')
    plt.show()
    plt.gcf().clear()

    discriminator(gen_not_ecgs)
    results = discriminator.getLatentRes()
    results = results.flatten()
    num_bins = 500
    n, bins, patches = plt.hist(results, num_bins, facecolor='blue')
    plt.show()
