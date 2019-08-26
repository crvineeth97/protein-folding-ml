import logging
from datetime import timedelta
from sys import argv
from time import time

import torch

from constants import DEVICE, LEARNING_RATE
from utils import get_model_dir, init_output_dir

# from models.lstm import LSTM
# from models.unet_1d import UNet
from models.resnet import ResNet
from preprocess import preprocess_raw_data
from test import test_model
from train import train_model

if len(argv) != 2:
    print("Please provide a description for the model and changes made")
    exit(1)

# Initialize model and logging
model = ResNet().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
init_output_dir(model)

# Preprocess raw ProteinNet data
start = time()
preprocess_raw_data()
preprocess_time = str(timedelta(seconds=time() - start))
logging.info("Total preprocessing time: %s", preprocess_time)

# Model Training
start = time()
train_model(model, criterion, optimizer)
train_time = str(timedelta(seconds=time() - start))
logging.info("Total training time: %s", train_time)

with open(get_model_dir() + "summary.txt", "a") as f:
    f.write("Model Description and changes: " + argv[1] + "\n")
    f.write("Training time: " + train_time + "\n")

# Model testing
test_model(model, criterion)
