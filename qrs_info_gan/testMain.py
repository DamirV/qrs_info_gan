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

a = torch.zeros([1, 1 , 5])
print(a)
print(a.shape)

a.squeeze_(0)
a.squeeze_(0)
print(a)
print(a.shape)
