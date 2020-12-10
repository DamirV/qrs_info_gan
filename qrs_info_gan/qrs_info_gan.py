from torch import nn
import torch.nn.functional as F
import torch
import data_loader

class Generator(nn.Module):
    def __init__(self, entry=8, radius=32):
        super(Generator, self).__init__()
        self.neuronCount1 = 2 * radius + 1
        self.neuronCount2 = self.neuronCount1 * 4
        self.model = nn.Sequential(
            nn.Linear(entry, self.neuronCount1),
            nn.ReLU(),
            nn.Linear(self.neuronCount1, self.neuronCount2),
            nn.ReLU(),
            nn.Linear(self.neuronCount2, self.neuronCount1))

    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, radius=32):
        super(Discriminator, self).__init__()
        self.neuronCount1 = 2 * radius + 1
        self.neuronCount2 = self.neuronCount1 // 2
        self.neuronCount3 = self.neuronCount2 // 4
        self.model = nn.Sequential(
            nn.Linear(self.neuronCount1, self.neuronCount2),
            nn.ReLU(),
            nn.Linear(self.neuronCount2, self.neuronCount3),
            nn.ReLU(),
            nn.Linear(self.neuronCount3, 1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "main":

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
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
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)