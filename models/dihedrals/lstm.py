import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import Base


class LSTM(Base):
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
        self.lstm = torch.nn.LSTM(
            input_size=self.num_input_dims,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(
            in_features=self.num_lstm_units, out_features=self.num_out_dims
        )
        self.act1 = torch.nn.Tanh()
        # self.fc2 = nn.Linear(in_features=512, out_features=256)
        # self.act2 = nn.Tanh()
        # self.fc3 = nn.Linear(in_features=256, out_features=self.num_out_dims)
        # self.act3 = nn.Tanh()

    def init_hidden(self, batch_size):
        # the weights are of the form (num_lstm_layers, batch_size, num_lstm_units)
        hidden_a = torch.randn(self.num_lstm_layers, batch_size, self.num_lstm_units)
        hidden_b = torch.randn(self.num_lstm_layers, batch_size, self.num_lstm_units)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input, lengths):
        self.hidden = self.init_hidden(len(lengths))
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
        output = output.view(-1, lengths[0], self.num_out_dims)
        output = output.transpose(1, 2)

        return output
