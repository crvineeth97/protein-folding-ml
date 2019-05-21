# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# seed random generator for reproducibility
torch.manual_seed(42)

# sample model borrowed from
# https://github.com/lblaabjerg/Master/blob/master/Models%20and%20processed%20data/ProteinNet_LSTM_500.py


class ResnetModel(nn.Module):
    def __init__(self,):
        super(ResnetModel, self).__init__()


class LSTMModel(nn.Module):
    def __init__(self, input_dims=20, num_lstms=2, lstm_dims=512, out_dims=3):
        super(LSTMModel, self).__init__()
        self.brnn = torch.nn.LSTM(
            input_size=input_dims,
            hidden_size=lstm_dims,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = torch.nn.Linear(in_features=2 * lstm_dims, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=out_dims)

    def generate_input(self, primary, evolutionary, lengths, embedding="one_hot"):
        """
        primary is of shape [minibatch_size, MAX_SEQ_LEN]
        """
        if embedding == "one_hot":
            pass

    def forward(self, padded_input, input_lengths):
        # padded_input is of size [Max_len, Batch_size, fv]
        output = pack_padded_sequence(padded_input, input_lengths, batch_first=True)
        output, _ = self.brnn(output)
        # Can get output.batch_sizes from our labels for training
        # But need the batch_sizes for prediction
        batch_sizes = output.batch_sizes
        output = nn.functional.relu(self.fc1(output.data))
        output = nn.functional.relu(self.fc2(output))
        output = nn.functional.softmax(self.fc3(output), dim=1)
        return output, batch_sizes
