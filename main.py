import logging
from datetime import timedelta
from sys import argv
from time import time

import torch

from constants import DEVICE, LEARNING_RATE

# from models.resnet import ResNet
# from models.lstm import LSTM
from models.unet import UNet
from preprocessing import preprocess_raw_data
from testing import test_model
from training import train_model
from util import get_model_dir, init_output_dir

if len(argv) != 2:
    print("Please provide a description for the model and changes made")
    exit(1)

# Initialize model and logging
model = UNet().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
init_output_dir(model)
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("Total number of parameters: %d", total_params)
logging.info("Number of trainable parameters: %d", train_params)

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
    f.write("Training time: " + train_time + "\n")
    f.write("Total number of parameters: " + str(total_params) + "\n")
    f.write("Number of trainable parameters: " + str(train_params) + "\n")
    f.write("Model Description and changes: " + argv[1] + "\n")

# Model testing
test_model(model, criterion)
