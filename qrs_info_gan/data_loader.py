import json
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


goodECG = '60757195'  #60757195 is good


def cutSignal(signal, start, radius):
    signal = signal[start-radius:start + radius]
    return signal


def completeSignal(signal):
    startLen = 5000 // 2 - len(signal) // 2
    startSignal = np.zeros(startLen)
    tempSignal = np.concatenate((startSignal, signal), axis=0)
    endLen = 5000 - len(tempSignal)
    endSignal = np.zeros(endLen)
    signal = np.concatenate((tempSignal, endSignal), axis=0)

    return signal


class QrsDataset(Dataset):
    def __init__(self, radius=32):
        self.data = self.load_data()
        self.testValue = self.data[goodECG]
        del self.data[goodECG]
        self.radius = radius
        self.len = 0
        for key in self.data:
            self.len += len(self.data[key]['Leads']['i']['Delineation']['qrs'])
            self.len -= 2

        self.keys = list(self.data.keys())
        self.keyIter = 0
        self.delIter = 0


    def load_data(self):
        PATH = "D:\\Projects\\ecg_gan_experiments-master\\Dataset\\"
        PATH2 = "/home/a/PycharmProjects/qrs_master/dataset/"
        FILENAME = "ecg_data_200.json"
        json_file = PATH + FILENAME
        with open(json_file, 'r') as f:
            data = json.load(f)

        return data


    def __getitem__(self, id):

        signal = self.data[self.keys[self.keyIter]]['Leads']['i']['Signal']
        deliniation = self.data[self.keys[self.keyIter]]['Leads']['i']['Delineation']['qrs']

        center = deliniation[self.delIter][1]

        if (center - self.radius < 0) or (center + self.radius > 5000):
            center = deliniation[3][1]

        signal = signal[center - self.radius:center + self.radius]
        signal = np.asarray(signal, dtype=np.float32)

        signal = torch.from_numpy(signal)
        self.delIter += 1
        if(self.delIter == len(deliniation)):
            self.delIter = 0
            self.keyIter += 1
            if(self.keyIter == len(self.keys)):
                self.keyIter = 0

        signal = signal.unsqueeze(0)

        return signal


    def __len__(self):
        return self.len


    def randCenter(self, deliniation, signal):
        center = random.randint(self.radius, 5000 - self.radius)

        for i in deliniation:
            intersection = (center > (i[1] - 2*self.radius)) and (center < (i[1] + 2*self.radius))
            if(intersection):
                center = self.randCenter(deliniation, signal)
                break

        return center


    def getTestSignal(self):
        signal = self.testValue['Leads']['i']['Signal']
        signal = np.asarray(signal, dtype=np.float32)
        return signal


    def getTestCenter(self):
        signal = self.testValue['Leads']['i']['Signal']
        deliniation = self.testValue['Leads']['i']['Delineation']['qrs']
        center = random.choice(deliniation)[1]
        signal = signal[center - self.radius:center + self.radius]
        return signal
