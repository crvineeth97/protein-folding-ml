from os import listdir

from numpy import load
from torch.utils.data import DataLoader, Dataset

from constants import MINIBATCH_SIZE


def contruct_dataloader_from_disk(foldername):
    return DataLoader(
        ProteinNetDataset(foldername),
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        collate_fn=merge_samples_to_minibatch,
        drop_last=True,
    )


class ProteinNetDataset(Dataset):
    def __init__(self, foldername):
        super(ProteinNetDataset, self).__init__()
        self.foldername = foldername
        self.filenames = listdir(foldername)

    def __getitem__(self, index):
        protein = load(self.foldername + self.filenames[index])
        length = protein["primary"].shape[0]
        if self.foldername[:5] == "valid" or self.foldername[:4] == "test":
            return (
                length,
                protein["primary"],
                protein["evolutionary"],
                protein["phi"],
                protein["psi"],
                protein["omega"],
                protein["tertiary"],
            )
        else:
            return (
                length,
                protein["primary"],
                protein["evolutionary"],
                protein["phi"],
                protein["psi"],
                protein["omega"],
            )

    def __len__(self):
        return len(self.filenames)


def merge_samples_to_minibatch(samples):
    # samples is a list of tuples and is of size MINIBATCH_SIZE
    # The tuples are the one returned from __getitem__ function
    # Therefore, x[0] will be the length of the protein
    # Sort the samples in decreasing order of their length
    # Can use the lengths and the specific order to generate the input fvs
    samples.sort(key=lambda x: x[0], reverse=True)
    return zip(*samples)
