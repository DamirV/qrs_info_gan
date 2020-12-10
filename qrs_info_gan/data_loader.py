import json
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


goodECG = '60757195'  #60757195 is good

class QrsDataset(Dataset):
    def __init__(self, radius=32):
        self.data = self.load_data()
        self.testValue = self.data[goodECG]
        del self.data[goodECG]
        self.radius = radius


    def load_data(self):
        PATH = "D:\\Projects\\ecg_gan_experiments-master\\Dataset\\"
        PATH2 = "/home/a/PycharmProjects/qrs_master/dataset/"
        FILENAME = "ecg_data_200.json"
        json_file = PATH + FILENAME
        with open(json_file, 'r') as f:
            data = json.load(f)

        return data


    def __getitem__(self, id):
        key, value = random.choice(list(self.data.items()))
        signal = value['Leads']['i']['Signal']

        deliniation = value['Leads']['i']['Delineation']['qrs']
        randomFlag = random.randint(0, 1)
        randomFlag = 1 # for gan

        if(randomFlag == 1):
            isqrs = 1
            center = random.choice(deliniation)[1]
        else:
            isqrs = 0
            center = self.randCenter(deliniation, signal)

        signal = signal[center - self.radius:center + self.radius + 1]

        signal = np.asarray(signal, dtype=np.float32)
        isqrs = np.asarray(isqrs, dtype=np.float32)

        signal = torch.from_numpy(signal)
        isqrs = torch.from_numpy(isqrs)

        return signal, isqrs


    def __len__(self):
        return 600 # заглушка


    def randCenter(self, deliniation, signal):
        center = random.randint(self.radius, 5000 - self.radius - 1)

        for i in deliniation:
            intersection = (center > (i[1] - 2*self.radius)) and (center < (i[1] + 2*self.radius + 1))
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
        signal = signal[center - self.radius:center + self.radius + 1]
        return signal
