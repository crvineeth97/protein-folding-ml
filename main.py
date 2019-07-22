import time

import torch

from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from training import train_model
from visualization import start_visualization

start_visualization()

device = torch.device("cpu")
if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = torch.device("cuda")

start = time.time()
preprocess_raw_data()
end = time.time()
print("Total preprocessing time: ", end - start)

model = ResNet(device).to(device)
start = time.time()
train_model_path = train_model(model, device)
end = time.time()
print("Total training time: ", end - start)
