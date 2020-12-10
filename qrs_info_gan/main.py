import torch
from torch import nn
import matplotlib as plt
import data_loader
import drawer
import qrs_info_gan
import math
import numpy as np


def saveModel(model, name):
    torch.save(model.state_dict(), f"D:/Projects/qrs_info_gan/model/{name}.pt")


def loadModel(name):
    model = torch.load(f"D:/Projects/qrs_info_gan/model/{name}.pt")
    model.eval()
    return model

def cutSignal(signal, start, radius):
    signal = signal[start-radius:start+radius+1]
    return signal


torch.manual_seed(111)
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    
discriminator = qrs_info_gan.Discriminator()
generator = qrs_info_gan.Generator()

radius=32
lr = 0.001
num_epochs = 10
loss_function = nn.BCELoss()
entry = 8

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

batch_size = 1
dataset = data_loader.QrsDataset()
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    for i, (real_samples, _) in enumerate(train_loader):
        real_samples_labels = torch.ones((batch_size, 1))
        generated_samples_labels = torch.zeros((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, entry))
        generated_samples = generator(latent_space_samples)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        #training discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        #training generator
        latent_space_samples = torch.randn((batch_size, entry))
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")
        print(f"Iteration: {i}")


print("success")

saveModel(generator, "generator")
saveModel(discriminator, "discriminator")

drr = drawer.Drawer(3)

signal = dataset.getTestCenter()
drr.add(signal)

inputData = torch.zeros(entry)
outputData = generator(inputData)
outputData = outputData.detach().numpy()
drr.add(outputData)


inputData = torch.zeros(entry)
inputData[0] = 1000
inputData[1] = -500
inputData[2] = 100
inputData[3] = 1

outputData = generator(inputData)
outputData = outputData.detach().numpy()
drr.add(outputData)

"""testSignal = dataset.getTestSignal()
drr.add(testSignal)

result = []
for i in range(radius, 5000-radius-1):
    signal = cutSignal(testSignal, i, radius)
    signal = torch.from_numpy(signal)
    tempResult = discriminator(signal)
    tempResult = tempResult.detach().numpy()
    result.append(tempResult[0])


drr.add(result)"""

drr.show()