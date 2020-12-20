import pickle


def save(pr):
    with open("file.pkl", "wb") as fp:
        pickle.dump(pr, fp)


def load():
    with open("file.pkl", "rb") as fp:
        return pickle.load(fp)


class Params:
    def __init__(self, radius=16, entry=8,
                 num_epochs=12, lr=0.001, batch_size=1):
        self.radius = radius
        self.entry = entry
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size


    def setRadius(self, radius):
        self.radius = radius


    def setEntry(self, entry):
        self.entry = entry


    def setNum_epochs(self, num_epochs):
        self.num_epochs = num_epochs


    def setLr(self, lr):
        self.lr = lr


    def setBatch_size(self, batch_size):
        self.batch_size = batch_size


    def __getstate__(self) -> dict:
        state = {}
        state["radius"] = self.radius
        state["entry"] = self.entry
        state["num_epochs"] = self.num_epochs
        state["lr"] = self.lr
        state["batch_size"] = self.batch_size
        return state

    def __setstate__(self, state: dict):  # Как мы будем восстанавливать класс из байтов
        self.radius = state["radius"]
        self.entry = state["entry"]
        self.num_epochs = state["num_epochs"]
        self.lr = state["lr"]
        self.batch_size = state["batch_size"]


