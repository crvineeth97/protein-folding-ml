from os import listdir

import numpy as np
from torch.utils.data import DataLoader, Dataset

from constants import MINIBATCH_SIZE


def contruct_dataloader_from_disk(
    foldername, set_name="dihedrals", batch_size=MINIBATCH_SIZE
):
    wanted_data = ["id", "primary", "evolutionary"]
    if set_name == "contact_map":
        wanted_data.extend(["contact_map"])
    else:
        wanted_data.extend(["phi", "psi", "omega"])
    return DataLoader(
        dataset=ProteinNet(foldername, wanted_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=merge_samples_to_minibatch,
        drop_last=False,
    )


class ProteinNet(Dataset):
    def __init__(self, foldername, wanted_data):
        super(ProteinNet, self).__init__()
        self.foldername = foldername
        self.wanted_data = wanted_data
        self.filenames = listdir(foldername)

    def __getitem__(self, index):
        protein = np.load(self.foldername + self.filenames[index])
        sample = dict((k, protein[k]) for k in self.wanted_data if k in protein)
        sample["length"] = protein["primary"].shape[0]
        if "validation" in self.foldername or "testing" in self.foldername:
            sample["tertiary"] = protein["tertiary"]
        return sample

    def __len__(self):
        return len(self.filenames)


def merge_samples_to_minibatch(samples):
    # samples is a list of dictionaries and is of size MINIBATCH_SIZE
    # The dicts are the one returned from __getitem__ function
    # Sort the samples in decreasing order of their length
    samples.sort(key=lambda x: x["length"], reverse=True)
    # Convert a list of dictionaries into a dictionary of lists
    batch = {k: [dic[k] for dic in samples] for k in samples[0]}
    return batch
