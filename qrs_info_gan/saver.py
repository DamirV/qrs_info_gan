import numpy as np
import torch


class Saver:
    def save(self, model, name):
        torch.save(model.state_dict(), f"D:/Projects/qrs_info_gan/model/{name}.pt")


    def load(self, model, name):
        model.load_state_dict(torch.load(f"D:/Projects/qrs_info_gan/model/{name}.pt"))
        model.eval()
        return model