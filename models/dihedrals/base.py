import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import pnerf.pnerf as pnerf
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
from utils import calculate_rmsd
from visualize import RamachandranPlot


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.init_model_vars()
        self.init_output_dir()
        self.init_logging()

    def init_model_vars(self):
        self.val_loader = contruct_dataloader_from_disk(VAL_FOLDER, "dihedrals")
        self.test_loader = contruct_dataloader_from_disk(TEST_FOLDER, "dihedrals")
        self.train_loader = contruct_dataloader_from_disk(TRAIN_FOLDER, "dihedrals")
        self.val_size = self.val_loader.__len__()
        self.test_size = self.test_loader.__len__()
        self.train_size = self.train_loader.__len__()
        self.val_last_batch_size = self.val_size % MINIBATCH_SIZE
        self.test_last_batch_size = self.test_size % MINIBATCH_SIZE
        self.train_last_batch_size = self.train_size % MINIBATCH_SIZE
        self.smallest_val_loss = 1e5
        self.best_model_rmsd = 1e5
        self.smallest_rmsd = 1e5
        self.epoch_iter = 0

    def init_output_dir(self):
        self.model_name = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.model_dir = "output/di/" + self.model_name + "/"
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
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        else:
            self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=LEARNING_RATE * 0.25, max_lr=LEARNING_RATE * 2)
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

        def transform_tertiary():
            tertiary = data["tertiary"]
            trans_tert = torch.zeros(
                batch_size, lengths[0] * 3, 3, device=DEVICE, dtype=torch.float32
            )
            for i in range(batch_size):
                for j in range(lengths[i]):
                    trans_tert[i, 3 * j + 0] = torch.from_numpy(tertiary[i][j, 0:3])
                    trans_tert[i, 3 * j + 1] = torch.from_numpy(tertiary[i][j, 3:6])
                    trans_tert[i, 3 * j + 2] = torch.from_numpy(tertiary[i][j, 6:9])
            return trans_tert

        def get_pred_coords_from_dihedrals():
            # The following will be of size [Batch, Length]
            pred_phi = torch.atan2(output[:, 0, :], output[:, 1, :]).unsqueeze(1)
            pred_psi = torch.atan2(output[:, 2, :], output[:, 3, :]).unsqueeze(1)
            # pred_omega = torch.atan2(output[:, 4, :], output[:, 5, :]).unsqueeze(1)
            pred_omega = torch.zeros(
                batch_size, 1, lengths[0], device=DEVICE, dtype=torch.float32
            )
            for j in range(batch_size):
                pred_omega[j, 0, : lengths[j]] = torch.from_numpy(data["omega"][j])

            # dihedrals will be converted from [Batch, 3, length]
            # [Length, Batch, 3] as this is the input
            # required by pnerf functions
            dihedrals = torch.cat((pred_phi, pred_psi, pred_omega), 1).permute(2, 0, 1)

            # Pnerf takes dihedrals and optional bond lengths and bond angles as input
            # and builds the coordinates so that rmsd can be calculated for loss
            # Divide by 100 to get in angstrom units
            # Coordinates will be of size [Length * 3, Batch, 3]
            # Last dimension represents the x, y, z coordinates
            # And the Length * 3 represents N, C_a, C for each AA
            coordinates = (
                pnerf.point_to_coordinate(
                    pnerf.dihedral_to_point(dihedrals, DEVICE), DEVICE
                )
                / 100
            )
            predicted_coords = coordinates.transpose(0, 1)
            return predicted_coords

        with torch.no_grad():
            running_loss = 0.0
            running_rmsd = 0.0
            for _, data in enumerate(self.val_loader):
                lengths = data["length"]
                batch_size = len(lengths)
                inp = self.generate_input(data)
                target = self.generate_target(data)
                output = self(inp, lengths)
                batch_loss = self.calculate_loss(lengths, output, target).item()
                running_loss += batch_loss
                pred_coords = get_pred_coords_from_dihedrals()
                act_coords = transform_tertiary()
                batch_rmsd = calculate_rmsd(lengths, pred_coords, act_coords)
                running_rmsd += batch_rmsd
            running_loss = exact_metric(running_loss, batch_loss)
            running_rmsd = exact_metric(running_rmsd, batch_rmsd)
        return running_loss, running_rmsd

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
                f.write("Best model RMSD: " + str(self.best_model_rmsd) + "\n")
                f.write("Smallest RMSD: " + str(self.smallest_rmsd) + "\n")

        visualize = None
        no_of_batches = self.train_size / MINIBATCH_SIZE
        if self.train_last_batch_size != 0:
            no_of_batches += 1
        while self.epoch_iter < TRAIN_EPOCHS:
            epoch_train_loss = 0.0
            running_train_loss = 0.0
            for batch_iter, data in enumerate(self.train_loader):
                lengths = data["length"]
                # inp should be of shape [Batch, 41, Max_length]
                inp = self.generate_input(data)
                # target should be of shape [Batch, 4, Max_length]
                target = self.generate_target(data)
                # output should be of shape [Batch, 4, Max_length]
                # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
                output = self(inp, lengths)
                batch_loss = self.calculate_loss(lengths, output, target)
                epoch_train_loss += batch_loss.item()
                running_train_loss += batch_loss.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                if self.scheduler != None:
                    self.scheduler.step()
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
                    val_loss, rmsd = self.validate()
                    if rmsd < self.smallest_rmsd:
                        self.smallest_rmsd = rmsd
                    if val_loss < self.smallest_val_loss:
                        self.smallest_val_loss = val_loss
                        self.best_model_rmsd = rmsd
                        self.save("best")
                    if not HIDE_UI:
                        # Shows Ramachandran plot of the first protein in batch
                        output = output.detach().cpu().numpy()[0]
                        pred_phi = np.arctan2(output[0, :], output[1, :])
                        pred_psi = np.arctan2(output[2, :], output[3, :])
                        if not visualize:
                            visualize = RamachandranPlot()
                        visualize.plot_ramachandran(
                            pred_phi, pred_psi, data["phi"][0], data["psi"][0]
                        )
                    logging.info(
                        "\tValidation loss: %.10lf, RMSD: %.10lf", val_loss, rmsd
                    )
            epoch_train_loss = exact_metric(epoch_train_loss, batch_loss.item())
            logging.info("Epoch train loss: %.10lf", epoch_train_loss)
            self.save("latest")
            self.epoch_iter += 1
        write_summary()
        logging.info("Best model validation loss: %.10lf", self.smallest_val_loss)
        logging.info("Best model RMSD: %.10lf", self.best_model_rmsd)
        logging.info("Best RMSD: %.10lf", self.smallest_rmsd)

    def generate_input(self, data):
        """
        Generate input for each minibatch. Pad the input feature vectors
        so that the final input shape is [MINIBATCH_SIZE, 41, Max_length]
        """
        lengths = data["length"]
        primary = data["primary"]
        evolutionary = data["evolutionary"]
        batch_size = len(lengths)
        transformed_primary = torch.zeros(
            batch_size, 20, lengths[0], device=DEVICE, dtype=torch.float32
        )

        # TODO: Use pythonic way
        for i in range(batch_size):
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_evolutionary = torch.zeros(
            batch_size, 21, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(batch_size):
            transformed_evolutionary[i, :, : lengths[i]] = torch.from_numpy(
                evolutionary[i].T
            )
        # transformed_primary           [n, 20, L]
        # transformed_evolutionary      [n, 21, L]
        # output                        [n, 41, L]
        return torch.cat((transformed_primary, transformed_evolutionary), dim=1)

    def generate_target(self, data):
        # dihedrals are in radians
        lengths = data["length"]
        phi = data["phi"]
        psi = data["psi"]
        batch_size = len(lengths)
        target = torch.zeros(
            batch_size, 4, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(batch_size):
            ph = torch.from_numpy(phi[i])
            ps = torch.from_numpy(psi[i])
            # om = torch.from_numpy(omega[i])
            target[i, 0, : lengths[i]] = torch.sin(ph)
            target[i, 1, : lengths[i]] = torch.cos(ph)
            target[i, 2, : lengths[i]] = torch.sin(ps)
            target[i, 3, : lengths[i]] = torch.cos(ps)
            # target[i, 4, : lengths[i]] = torch.sin(om)
            # target[i, 5, : lengths[i]] = torch.cos(om)
        return target

    def calculate_loss(self, lengths, output, target):
        batch_size = len(lengths)
        loss = self.criterion(output[0], target[0])
        for i in range(1, batch_size):
            loss += self.criterion(
                output[i, :, : lengths[i]], target[i, :, : lengths[i]]
            )
        loss /= batch_size
        return loss
