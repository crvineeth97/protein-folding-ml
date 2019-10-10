import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset

from constants import MINIBATCH_SIZE, AA_ID_DICT


def contruct_dataloader_from_disk(
    filename, set_name="dihedrals", batch_size=MINIBATCH_SIZE
):
    wanted_data = ["name", "PSSM", "OtherPairs"]
    if set_name == "contact_map":
        wanted_data.extend(["contactMatrix"])
    else:
        wanted_data.extend(["phi", "psi", "omega"])
    return DataLoader(
        dataset=RaptorX(filename, wanted_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=merge_samples_to_minibatch,
        drop_last=False,
    )


class RaptorX(Dataset):
    def __init__(self, filename, wanted_data):
        super(RaptorX, self).__init__()
        with open(filename, "rb") as f:
            self.proteins = pickle.load(f, encoding="latin1")
        self.wanted_data = wanted_data

    def __getitem__(self, index):
        protein = self.proteins[index]
        sample = dict((k, protein[k]) for k in self.wanted_data if k in protein)
        sample["sequence"] = np.array([AA_ID_DICT[aa] for aa in protein["sequence"]])
        sample["length"] = sample["sequence"].shape[0]
        return sample

    def __len__(self):
        return len(self.proteins)


def merge_samples_to_minibatch(samples):
    # samples is a list of dictionaries and is of size MINIBATCH_SIZE
    # The dicts are the one returned from __getitem__ function
    # Sort the samples in decreasing order of their length
    samples.sort(key=lambda x: x["length"], reverse=True)
    # Convert a list of dictionaries into a dictionary of lists
    batch = {k: [dic[k] for dic in samples] for k in samples[0]}
    return batch

