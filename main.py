import datetime
import logging
from time import time
import sys

import torch

from constants import DEVICE, LEARNING_RATE
from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from testing import test_model
from training import train_model
from util import get_model_dir, init_output_dir

if len(sys.argv) != 2:
    print("Please provide a description for the model and changes made")
    exit(1)

model = ResNet().to(DEVICE)
criterion = torch.nn.MSELoss()
# TODO Try various options of Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
init_output_dir(model)
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("Total number of parameters: %d", total_params)
logging.info("Number of trainable parameters: %d", train_params)

start = time()
preprocess_raw_data()
logging.info("Total preprocessing time: %f", time() - start)

start = time()
train_model(model, criterion, optimizer)
train_time = str(datetime.timedelta(seconds=time() - start))
logging.info("Total training time: %s", train_time)

with open(get_model_dir() + "summary.txt", "a") as f:
    f.write("Training time: " + train_time)
    f.write("Total number of parameters: " + str(total_params) + "\n")
    f.write("Number of trainable parameters: " + str(train_params) + "\n")
    f.write("Model Description and changes: " + sys.argv[1] + "\n")

test_model(model, criterion)
