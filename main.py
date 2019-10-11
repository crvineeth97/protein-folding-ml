import logging
import sys

# import os
from datetime import timedelta
from time import time

from constants import DEVICE

# from gpu_profile import gpu_profile

# from models.dihedrals.resnet_1d import ResNet
from models.contact_map.unet_2d import UNet
from proteinnet_preprocess import preprocess_raw_data

# sys.settrace(gpu_profile)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["GPU_DEBUG"] = "0"

if len(sys.argv) != 2:
    print("Please provide a description for the model and changes made")
    exit(1)

# Initialize model
model = UNet().to(DEVICE)
model.set_criterion_and_optimizer()

# Preprocess raw ProteinNet data
start = time()
preprocess_raw_data()
preprocess_time = str(timedelta(seconds=time() - start))
logging.info("Total preprocessing time: %s", preprocess_time)

# Model Training
start = time()
model.train()
train_time = str(timedelta(seconds=time() - start))
logging.info("Total training time: %s", train_time)

with open(model.model_dir + "summary.txt", "a") as f:
    f.write("Model Description and changes: " + sys.argv[1] + "\n")
    f.write("Training time: " + train_time + "\n")
