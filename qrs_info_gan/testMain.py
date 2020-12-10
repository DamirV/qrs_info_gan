import numpy as np
import torch
from matplotlib import pyplot as plt
import drawer
import qrs_info_gan

def saveModel(model, name):
    torch.save(model.state_dict(), f"D:/Projects/qrs_info_gan/model/{name}.pt")


def loadModel(name):
    model = torch.load(f"D:/Projects/qrs_info_gan/model/{name}.pt")
    model.eval()
    return model


drr = drawer.Drawer(1)
generator = loadModel("generator")


inputData = torch.zeros(8)
outputData = generator(inputData)
outputData = outputData.detach().numpy()
drr.add(outputData)
