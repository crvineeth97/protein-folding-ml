from os import listdir

from numpy import load
from torch.utils.data import DataLoader, Dataset

from constants import MINIBATCH_SIZE


def contruct_dataloader_from_disk(foldername, device):
    return DataLoader(
        ProteinNetDataset(foldername, device),
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        collate_fn=merge_samples_to_minibatch,
        drop_last=True,
    )


class ProteinNetDataset(Dataset):
    def __init__(self, foldername, device):
        super(ProteinNetDataset, self).__init__()
        self.foldername = foldername
        self.filenames = listdir(foldername)
        self.device = device

    def __getitem__(self, index):
        protein = load(self.foldername + self.filenames[index])
        primary = protein["primary"]
        # primary = torch.tensor(
        #     protein["primary"], dtype=torch.uint8, device=self.device
        # )
        # evolutionary = torch.tensor(
        #     protein["evolutionary"], dtype=torch.float, device=self.device
        # )
        # secondary = torch.tensor(protein["secondary"], dtype=torch.uint8, device=self.device)
        # phi = torch.tensor(protein["phi"], dtype=torch.float, device=self.device)
        # psi = torch.tensor(protein["psi"], dtype=torch.float, device=self.device)
        length = primary.shape[0]
        return length, primary, protein["evolutionary"], protein["phi"], protein["psi"]

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
