import torch
from torch import nn
import matplotlib as plt
import data_loader
import drawer
import qrs_info_gan
import math
import numpy as np
import datetime

radius = 16
dataset = data_loader.QrsDataset(radius)
model = qrs_info_gan.Discriminator(radius)
now = datetime.datetime.now()

def cutSignal(signal, start, radius):
    signal = signal[start-radius:start+radius+1]
    return signal


model.load_state_dict(torch.load("D:\Projects\qrs_info_gan\model\discriminator.pt"))
model.eval()

drr = drawer.Drawer(2, now)

testSignal = dataset.getTestSignal()
drr.add(testSignal)

result = []
for i in range(radius, 5000-radius-1):
    signal = cutSignal(testSignal, i, radius)
    signal = torch.from_numpy(signal)
    tempResult = model(signal)
    tempResult = tempResult.detach().numpy()
    result.append(tempResult[0])


drr.add(result)

drr.save()
drr.show()





