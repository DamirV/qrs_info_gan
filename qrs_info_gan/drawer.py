import numpy as np
from matplotlib import pyplot as plt


def draw(y, number=0):
    x = np.arange(0, len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    fig.savefig(f'D:/Projects/qrs_info_gan/graphs/graph{number}.pdf')


class Drawer:
    def __init__(self, figures=1):
        self.fig = plt.figure()
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
        plt.show()


    def save(self):
        self.fig.savefig('D:/Projects/qrs_info_gan/graphs/graph.pdf')