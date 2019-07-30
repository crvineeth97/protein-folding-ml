from time import time

from constants import DEVICE
from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from training import train_model
from util import init_output_dir

start = time()
preprocess_raw_data()
print("Total preprocessing time: ", time() - start)

init_output_dir()

model = ResNet().to(DEVICE)
start = time()
train_model_path = train_model(model)
print("Total training time: ", time() - start)
