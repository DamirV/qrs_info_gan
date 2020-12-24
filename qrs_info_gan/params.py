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
    patch_len: int = 256  # size of ecg patch, need to be degree of 2
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

