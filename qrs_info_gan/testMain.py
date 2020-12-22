import torch
from torch import nn
import matplotlib as plt
import data_loader
import drawer
import qrs_info_gan
import math
import numpy as np
import datetime
import logging
import params

now = datetime.datetime.now()
path = drawer.makeDir(now)

pr = params.Params(radius=10, entry=8, num_epochs=20, lr=0.001, batch_size=1)

dataset = data_loader.QrsDataset(pr.radius)
discriminator, generator, disLoss, genLoss = qrs_info_gan.train(pr, dataset)

qrs_info_gan.saveModel(generator, "generator", path)
qrs_info_gan.saveModel(discriminator, "discriminator", path)

errDrr = drawer.ErrDrawer(path, 1)

print(disLoss)
print(genLoss)

errDrr.add(disLoss, "red", "dis")
errDrr.add(genLoss, "blue", "gen")

errDrr.save()
params.save(pr, path)

drr = drawer.SignalDrawer(4, path, 0)
drr.add(dataset.getTestCenter())

input = torch.zeros(pr.entry)
output = generator(input)
output = output.detach().numpy()
drr.add(output)

testSignal = dataset.getTestSignal()
drr.add(testSignal)

result = []
for i in range(pr.radius, 5000-pr.radius-1):
    signal = data_loader.cutSignal(testSignal, i, pr.radius)
    signal = torch.from_numpy(signal)
    tempResult = discriminator(signal)
    tempResult = tempResult.detach().numpy()
    result.append(tempResult[0])


drr.add(result)

drr.save()
