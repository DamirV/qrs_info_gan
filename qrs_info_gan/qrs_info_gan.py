import datetime
import logging
from torch import nn
import torch.nn.functional as F
import torch
import data_loader
import drawer
import params


def saveModel(model, name, path):
        torch.save(model.state_dict(), f"{path}/{name}.pt")


def loadModel(model, name, path):
        model.load_state_dict(torch.load(f"{path}/{name}.pt"))
        model.eval()
        return model


class Generator(nn.Module):
    def __init__(self, entry, radius): #entry=8, radius=32
        super(Generator, self).__init__()
        self.neuronCount1 = (2 * radius + 1) * 4
        self.neuronCount2 = self.neuronCount1
        self.neuronCount3 = self.neuronCount1 // 2

        self.model = nn.Sequential(
            nn.Linear(entry, self.neuronCount1),
            nn.ReLU(),
            nn.Linear(self.neuronCount1, self.neuronCount2),
            nn.ReLU(),
            nn.Linear(self.neuronCount2, self.neuronCount3),
            nn.ReLU(),
            nn.Linear(self.neuronCount3, 2 * radius + 1))


    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, radius):
        super(Discriminator, self).__init__()
        self.neuronCount1 = 2 * radius + 1
        self.neuronCount2 = self.neuronCount1 // 2
        self.neuronCount3 = self.neuronCount2 // 2
        self.neuronCount4 = self.neuronCount2 // 2

        self.model = nn.Sequential(
            nn.Linear(self.neuronCount1, self.neuronCount2),
            nn.ReLU(),
            nn.Linear(self.neuronCount2, self.neuronCount3),
            nn.ReLU(),
            nn.Linear(self.neuronCount3, self.neuronCount4),
            nn.ReLU(),
            nn.Linear(self.neuronCount4, 1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.model(x)
        return x


def train(pr, dataset):
    torch.manual_seed(111)
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    discriminator = Discriminator(pr.radius)
    generator = Generator(pr.entry, pr.radius)

    loss_function = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=pr.lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=pr.lr)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=pr.batch_size, shuffle=True)

    disLoss = []
    genLoss = []
    for epoch in range(pr.num_epochs):

        for i, (real_samples, _) in enumerate(train_loader):
            real_samples_labels = torch.ones((pr.batch_size, 1))
            generated_samples_labels = torch.zeros((pr.batch_size, 1))
            latent_space_samples = torch.randn((pr.batch_size, pr.entry))
            generated_samples = generator(latent_space_samples)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # training discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # training generator
            latent_space_samples = torch.randn((pr.batch_size, pr.entry))
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            print(f"Iteration: {i}")

            if i == train_loader.__len__() - 1:
                genLoss.append(loss_generator.item())
                disLoss.append(loss_discriminator.item())

    print("success")
    return discriminator, generator, disLoss, genLoss


def experiment(pr):
    now = datetime.datetime.now()
    path = drawer.makeDir(now)

    dataset = data_loader.QrsDataset(pr.radius)
    discriminator, generator, disLoss, genLoss = train(pr, dataset)

    saveModel(generator, "generator", path)
    saveModel(discriminator, "discriminator", path)

    errDrr = drawer.ErrDrawer(path, 1)

    errDrr.add(disLoss, "red", "dis")
    errDrr.add(genLoss, "blue", "gen")

    errDrr.save()
    params.save(pr, path)
    pr.save(path)

    drr = drawer.SignalDrawer(4, path, 0)
    drr.add(dataset.getTestCenter())

    input = torch.zeros(pr.entry)
    output = generator(input)
    output = output.detach().numpy()
    drr.add(output)

    testSignal = dataset.getTestSignal()
    drr.add(testSignal)

    result = []
    for i in range(pr.radius, 5000 - pr.radius - 1):
        signal = data_loader.cutSignal(testSignal, i, pr.radius)
        signal = torch.from_numpy(signal)
        tempResult = discriminator(signal)
        tempResult = tempResult.detach().numpy()
        result.append(tempResult[0])

    drr.add(result)
    drr.save()
    drr.clear()
