import torch
from torch import nn
import matplotlib as plt
import data_loader
import drawer
import qrs_info_gan
import math
import numpy as np
import datetime

radius = 32

dataset = data_loader.QrsDataset(radius)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True)

now = datetime.datetime.now()

drr = drawer.Drawer(5, now)

for epoch in range(1):
    for i, (real_samples, _) in enumerate(train_loader):
        a = real_samples[0]
        a = a.detach().numpy()
        print(a)
        drr.add(a)
        if i == 5:
            break

drr.show()
