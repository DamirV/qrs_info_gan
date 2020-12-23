import pickle
from typing import NamedTuple

def save(pr, path):
    serialized = pickle.dumps(pr)
    with open(f"{path}/params.pkl", "wb") as fp:
        pickle.dump(serialized, fp)


def load(path):
    with open(f"{path}/params.pkl", "rb") as fp:
        serialized = pickle.load(fp)
        return pickle.loads(serialized)


class Params(NamedTuple):
    experiment_folder: str

    radius: int = 128
    entry: int = 8
    num_epochs: int = 12
    # params of data generator
    step_size: int = 25  # num discrets in one step
    max_steps_left: int = 2  # num steps from patch center, allowed for the moving complex
    n_classes: int = 5  # number of classes for dataset
    patch_len: int = radius*2 + 1  # size of ecg patch, need to be degree of 2
    num_channels: int = 1  # "number of channels in ecg, no more than 12"

    # params of model
    code_dim: int = 2
    latent_dim: int = 36

    # params of training
    lr: float = 0.0002  # adam: learning rate
    b1: float = 0.5
    b2: float = 0.999  # adam: decay of first order momentum of gradient
    batch_size: int = 12
    n_epochs: int = 2501

    # params of logger
    save_pic_interval: int = 30  # in epoches
    save_model_interval: int = 30  # in epoches


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

    def toStr(self):
        str = f"radius = {self.radius}\n"
        str = str + f"entry = {self.entry}\n"
        str = str + f"num_epochs = {self.num_epochs}\n"
        str = str + f"lr = {self.lr}\n"
        str = str + f"batch_size = {self.batch_size}\n"
        return str

    def save(self, path):
        my_file = open(f"{path}/params.txt", "w")
        my_file.write(self.toStr())
        my_file.close()
