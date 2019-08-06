import logging
from time import time
import torch
from constants import DEVICE, LEARNING_RATE
from models.resnet import ResNet
from preprocessing import preprocess_raw_data
from training import train_model
from testing import test_model
from util import init_output_dir

model = ResNet().to(DEVICE)
criterion = torch.nn.MSELoss()
# TODO Try various options of Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
init_output_dir(model)

start = time()
preprocess_raw_data()
logging.info("Total preprocessing time: %f", time() - start)

start = time()
train_model(model, criterion, optimizer)
logging.info("Total training time: %f", time() - start)

start = time()
test_model(model, criterion)
logging.info("Total testing time: %f", time() - start)
