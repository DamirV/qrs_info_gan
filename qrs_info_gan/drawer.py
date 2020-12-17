import numpy as np
from matplotlib import pyplot as plt
import os


def draw(y, number=0):
    x = np.arange(0, len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    fig.savefig(f'D:/Projects/qrs_info_gan/graphs/graph{number}.pdf')


class Drawer:
    def __init__(self, figures, date):
        self.fig = plt.figure()
        self.date = date
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


    def show(self):
        #self.makeDir()
        #self.fig.savefig(f'D:/Projects/qrs_info_gan/graphs/{self.strDate}/graph.pdf')
        plt.show()


    def save(self):
        self.makeDir()
        self.fig.savefig(f'D:/Projects/qrs_info_gan/graphs/{self.strDate}/graph.pdf')


    def makeDir(self):
        os.chdir("graphs")
        self.strDate = str(f"{self.date.year}-{self.date.month}-{self.date.day}--{self.date.hour}-{self.date.minute}-{self.date.second}")
        os.mkdir(self.strDate)



