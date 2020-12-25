import numpy as np
from matplotlib import pyplot as plt
import os


def makeDir(date):
    strDate = str(f"{date.year}-{date.month}-{date.day}--{date.hour}-{date.minute}-{date.second}")
    os.mkdir(strDate)
    return f'D:/Projects/qrs_info_gan/graphs/{strDate}'


class SignalDrawer:
    def __init__(self, figures, path, number):
        self.fig = plt.figure(number, figsize=(8, 6))
        self.path = path
        self.count = 0
        self.figures = figures
        self.axs = []
        for i in range(figures):
            self.axs.append(self.fig.add_subplot(figures, 1, i+1))


    def add(self, y):
        if self.count == self.figures:
            return

        x = np.arange(0, len(y))
        self.axs[self.count].plot(x, y)
        self.count += 1


    def save(self):
        self.fig.savefig(f"{self.path}/graph.pdf")


    def save2(self):
        self.fig.savefig(f"{self.path}/graph2.pdf")


    def show(self):
        plt.show()


    def clear(self):
        plt.gcf().clear()

class ErrDrawer:
    def __init__(self, path, number):
        self.fig, self.ax = plt.subplots(number, figsize=(8, 6))
        self.path = path


    def add(self, y, color, label):
        x = np.arange(0, len(y))
        self.ax.plot(x, y, color=color, label=label)


    def save(self):
        self.fig.savefig(f"{self.path}/loss.pdf")


    def show(self):
        plt.show()


