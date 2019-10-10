import logging
import sys
from datetime import timedelta
from time import time

from constants import DEVICE

# from models.dihedrals.resnet_1d import ResNet
from models.contact_map.unet_2d import UNet

if len(sys.argv) != 2:
    print("Please provide a description for the model and changes made")
    exit(1)

# Initialize model
model = UNet().to(DEVICE)
model.set_criterion_and_optimizer()

# Model Training
start = time()
model.train()
train_time = str(timedelta(seconds=time() - start))
logging.info("Total training time: %s", train_time)

with open(model.model_dir + "summary.txt", "a") as f:
    f.write("Model Description and changes: " + sys.argv[1] + "\n")
    f.write("Training time: " + train_time + "\n")
