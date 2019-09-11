import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from constants import (
    DEVICE,
    EVAL_INTERVAL,
    HIDE_UI,
    LEARNING_RATE,
    MINIBATCH_SIZE,
    PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES,
    PRINT_LOSS_INTERVAL,
    TEST_FOLDER,
    TRAIN_EPOCHS,
    TRAIN_FOLDER,
    VAL_FOLDER,
)
from dataloader import contruct_dataloader_from_disk
from visualize import ContactMap


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.init_model_vars()
        self.init_output_dir()
        self.init_logging()

    def init_model_vars(self):
        self.val_loader = contruct_dataloader_from_disk(VAL_FOLDER, "contact_map")
        self.test_loader = contruct_dataloader_from_disk(TEST_FOLDER, "contact_map")
        self.train_loader = contruct_dataloader_from_disk(TRAIN_FOLDER, "contact_map")
        self.val_size = self.val_loader.__len__()
        self.test_size = self.test_loader.__len__()
        self.train_size = self.train_loader.__len__()
        self.val_last_batch_size = self.val_size % MINIBATCH_SIZE
        self.test_last_batch_size = self.test_size % MINIBATCH_SIZE
        self.train_last_batch_size = self.train_size % MINIBATCH_SIZE
        self.smallest_val_loss = 1e5
        self.epoch_iter = 0

    def init_output_dir(self):
        self.model_name = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.model_dir = "output/cm/" + self.model_name + "/"
        os.makedirs(self.model_dir)
        os.makedirs(self.model_dir + "code/")
        # Keep a copy of .sh and .py files in the model's code
        os.system(
            "rsync -mar --exclude='output' --include='*/' "
            + "--include='*\.py' --include='*\.sh' --exclude='*' ./ "
            + "\ ".join(self.model_dir.split(" "))
            + "code/"
        )

    def init_logging(self):
        file_handler = logging.FileHandler(filename=self.model_dir + "log.txt")
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            handlers=handlers,
        )
        logging.info("DEVICE: %s", DEVICE)
        logging.info("EVAL_INTERVAL: %d", EVAL_INTERVAL)
        logging.info("LEARNING_RATE: %f", LEARNING_RATE)
        logging.info("MINIBATCH_SIZE: %d", MINIBATCH_SIZE)
        logging.info(
            "PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES: %s",
            str(PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES),
        )
        logging.info("PRINT_LOSS_INTERVAL: %d", PRINT_LOSS_INTERVAL)
        logging.info("TRAINING_EPOCHS: %d", TRAIN_EPOCHS)

    def set_criterion_and_optimizer(self, criterion=None, optimizer=None):
        if criterion is None:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = criterion
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        else:
            self.optimizer = optimizer
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Total number of parameters: %d", total_params)
        logging.info("Number of trainable parameters: %d", train_params)
        logging.info(self)

    def save(self, name):
        torch.save(self.state_dict(), self.model_dir + name + ".pth")

    def validate(self):
        def exact_metric(running, batch):
            return (
                (running - batch) * MINIBATCH_SIZE + batch * self.val_last_batch_size
            ) / self.val_size

        with torch.no_grad():
            running_loss = 0.0
            for _, data in enumerate(self.val_loader):
                lengths = data["length"]
                if lengths[0] < 32 or lengths[0] > 512:
                    continue
                inp = self.generate_input(data)
                target = self.generate_target(data)
                output = self(inp, lengths)
                batch_loss = self.calculate_loss(lengths, output, target).item()
                running_loss += batch_loss
            running_loss = exact_metric(running_loss, batch_loss)
        return running_loss

    def train(self):
        def exact_metric(running, batch):
            return (
                (running - batch) * MINIBATCH_SIZE + batch * self.train_last_batch_size
            ) / self.train_size

        def write_summary():
            with open(self.model_dir + "summary.txt", "a") as f:
                f.write("Number of epochs: " + str(self.epoch_iter) + "\n")
                f.write("Latest epoch train loss: " + str(epoch_train_loss) + "\n")
                f.write(
                    "Smallest validation loss: " + str(self.smallest_val_loss) + "\n"
                )

        visualize = None
        no_of_batches = self.train_size // MINIBATCH_SIZE
        if self.train_last_batch_size != 0:
            no_of_batches += 1
        while self.epoch_iter < TRAIN_EPOCHS:
            epoch_train_loss = 0.0
            running_train_loss = 0.0
            for batch_iter, data in enumerate(self.train_loader):
                lengths = data["length"]
                if lengths[0] < 32 or lengths[0] > 512:
                    continue
                # inp should be of shape [Batch, 123, Max_length, Max_length]
                inp = self.generate_input(data)
                # target should be of shape [Batch, Max_length, Max_length]
                target = self.generate_target(data)
                # output should be of shape [Batch, Max_length, Max_length]
                # Batches of contact maps
                output = self(inp, lengths)
                batch_loss = self.calculate_loss(lengths, output, target)
                epoch_train_loss += batch_loss.item()
                running_train_loss += batch_loss.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                if (batch_iter + 1) % PRINT_LOSS_INTERVAL == 0:
                    logging.info(
                        "[%d|%.2f%%] Train loss: %.10lf",
                        self.epoch_iter,
                        ((batch_iter + 1) / no_of_batches) * 100,
                        running_train_loss / PRINT_LOSS_INTERVAL,
                    )
                    running_train_loss = 0.0

                if batch_iter % EVAL_INTERVAL == 0:
                    # Finding validation loss in the beginning just to see how much
                    # the model improves after time
                    val_loss = self.validate()
                    if val_loss < self.smallest_val_loss:
                        self.smallest_val_loss = val_loss
                        self.save("best")
                    if not HIDE_UI:
                        # Plot contact map of first protein in batch
                        output = output.detach().cpu().numpy()[0]
                        if not visualize:
                            visualize = ContactMap()
                        visualize.plot_contact_map(data["id"], output)
                    logging.info("\tValidation loss: %.10lf", val_loss)
            epoch_train_loss = exact_metric(epoch_train_loss, batch_loss.item())
            logging.info("Epoch train loss: %.10lf", epoch_train_loss)
            self.save("latest")
            self.epoch_iter += 1
        write_summary()
        logging.info("Best model validation loss: %.10lf", self.smallest_val_loss)

    def generate_input(self, data):
        """
        Generate input for each minibatch. Pad the input feature vectors
        so that the final input shape is [MINIBATCH_SIZE, Channels, Max_length, Max_length]
        """
        lengths = data["length"]
        primary = data["primary"]
        evolutionary = data["evolutionary"]
        batch_size = len(lengths)
        inp = torch.zeros(
            batch_size, 41 * 3, lengths[0], lengths[0], dtype=torch.float32
        )

        feature_1d = torch.zeros(batch_size, 41, lengths[0], dtype=torch.float32)

        for i in range(batch_size):
            feature_1d[i, :20, : lengths[i]] = torch.from_numpy(
                np.eye(20)[primary[i]].T
            )
            feature_1d[i, 20:, : lengths[i]] = torch.from_numpy(evolutionary[i].T)

        for k in range(batch_size):
            for i in range(lengths[k]):
                for j in range(lengths[k]):
                    inp[k, :, i, j] = torch.cat(
                        (
                            feature_1d[k, :, i],
                            feature_1d[k, :, (i + j) // 2],
                            feature_1d[k, :, j],
                        ),
                        0,
                    )
        return inp.to(DEVICE)

    def generate_target(self, data):
        # dihedrals are in radians
        lengths = data["length"]
        contact_map = data["contact_map"]
        batch_size = len(lengths)
        target = torch.zeros(
            batch_size, lengths[0], lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(batch_size):
            target[i, : lengths[i], : lengths[i]] = torch.from_numpy(contact_map[i])
        return target

    def calculate_loss(self, lengths, output, target):
        batch_size = len(lengths)
        loss = self.criterion(output[0], target[0])
        for i in range(1, batch_size):
            loss += self.criterion(
                output[i, : lengths[i], : lengths[i]],
                target[i, : lengths[i], : lengths[i]],
            )
        loss /= batch_size
        return loss
