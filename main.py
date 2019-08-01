import logging
from time import time

from constants import DEVICE
from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from training import train_model
from util import init_output_dir

model = ResNet().to(DEVICE)
init_output_dir(model)

start = time()
preprocess_raw_data()
logging.info("Total preprocessing time: %f", time() - start)

start = time()
train_model_path = train_model(model)
logging.info("Total training time: %f", time() - start)
