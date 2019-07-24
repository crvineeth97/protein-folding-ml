import time

from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from training import train_model
from constants import DEVICE

start = time.time()
preprocess_raw_data()
end = time.time()
print("Total preprocessing time: ", end - start)

model = ResNet().to(DEVICE)
start = time.time()
train_model_path = train_model(model)
end = time.time()
print("Total training time: ", end - start)
