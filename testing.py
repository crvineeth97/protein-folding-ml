import torch


def load_model(path):
    model = torch.load(path)
    
    return model
