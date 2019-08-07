import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from constants import DEVICE, MINIBATCH_SIZE


class LSTM(nn.Module):
    def __init__(
        self, num_input_dims=41, num_lstm_units=100, num_lstm_layers=2, num_out_dims=4
    ):
        super(LSTM, self).__init__()
        self.num_input_dims = num_input_dims
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.num_out_dims = num_out_dims
        # build actual NN
        self.__build_model()

    def __build_model(self):
        # self.brnn = nn.LSTM(
        #     input_size=input_dims,
        #     hidden_size=lstm_dims,
        #     num_lstm_layers=2,
        #     bias=True,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.lstm = nn.LSTM(
            input_size=self.num_input_dims,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            in_features=self.num_lstm_units, out_features=self.num_out_dims
        )
        self.act1 = nn.Tanh()
        # self.fc2 = nn.Linear(in_features=512, out_features=256)
        # self.act2 = nn.Tanh()
        # self.fc3 = nn.Linear(in_features=256, out_features=self.num_out_dims)
        # self.act3 = nn.Tanh()

    def init_hidden(self):
        # the weights are of the form (num_lstm_layers, batch_size, num_lstm_units)
        hidden_a = torch.randn(
            self.num_lstm_layers, MINIBATCH_SIZE, self.num_lstm_units, device=DEVICE
        )
        hidden_b = torch.randn(
            self.num_lstm_layers, MINIBATCH_SIZE, self.num_lstm_units, device=DEVICE
        )

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input, lengths):
        self.hidden = self.init_hidden()
        # [Batch, 41, Max_length] -> [Batch, Max_length, 41]
        output = input.transpose(1, 2)

        # [Batch, Max_length, embedding_dim] -> [Batch, Max_length, num_lstm_units]
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        output = pack_padded_sequence(output, lengths, batch_first=True)
        # now run through LSTM
        output, self.hidden = self.lstm(output, self.hidden)

        # undo the packing operation
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Project to target space
        # [Batch, Max_length, num_lstm_units] -> [Batch * Max_length, num_lstm_units]
        output = output.contiguous().view(-1, self.num_lstm_units)

        # Run through linear and activation layers
        output = self.fc1(output)
        output = self.act1(output)

        # [Batch * Max_length, num_lstm_units] -> [Batch, num_out_dims, Max_length]
        output = output.view(MINIBATCH_SIZE, lengths[0], self.num_out_dims)
        output = output.transpose(1, 2)

        return output

    def generate_input(self, lengths, primary, evolutionary):
        """
        Generate input for each minibatch. Pad the input feature vectors
        so that the final input shape is [MINIBATCH_SIZE, 41, Max_length]
        Args:
        lengths: Tuple of all protein lengths in current minibatch
        primary: Tuple of numpy arrays of shape (l,) describing the
        protein amino acid sequence, which are of variable length
        evolutionary: Tuple of numpy arrays of shape (l, 21) describing
        the PSSM matrix of the protein
        """

        transformed_primary = torch.zeros(
            MINIBATCH_SIZE, 20, lengths[0], device=DEVICE, dtype=torch.float32
        )

        # TODO: Use pythonic way
        for i in range(MINIBATCH_SIZE):
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_evolutionary = torch.zeros(
            MINIBATCH_SIZE, 21, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(MINIBATCH_SIZE):
            transformed_evolutionary[i, :, : lengths[i]] = torch.from_numpy(
                evolutionary[i].T
            )
        # transformed_primary           [n, 20, L]
        # transformed_evolutionary      [n, 21, L]
        # output                        [n, 41, L]
        return torch.cat((transformed_primary, transformed_evolutionary), dim=1)

    def generate_target(self, lengths, phi, psi, omega):
        # dihedrals are in radians
        target = torch.zeros(
            MINIBATCH_SIZE, 4, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(MINIBATCH_SIZE):
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

    def calculate_loss(self, lengths, criterion, output, target):
        loss = criterion(output[0], target[0])
        for i in range(1, MINIBATCH_SIZE):
            loss += criterion(output[i, :, : lengths[i]], target[i, :, : lengths[i]])
        loss /= MINIBATCH_SIZE
        return loss
