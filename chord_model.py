
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from pathlib import Path
import os


PACKAGE_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
MODELS_DIR = PACKAGE_DIR / 'models'
CUDA_ENABLED = torch.cuda.is_available()


FloatTensor = torch.cuda.FloatTensor if CUDA_ENABLED else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA_ENABLED else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if CUDA_ENABLED else torch.ByteTensor
Tensor = FloatTensor


class ChordModel(nn.Module):

    def __init__(self, rnn_type, n_features, n_hidden, n_tokens, n_layers, dropout=0.75):
        super(ChordModel, self).__init__()

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_features, n_hidden, n_layers, dropout=dropout)
        else:
            try:
                non_linearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(""" An Invalid option for '--model' was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU]""")
            self.rnn = nn.RNN(n_features, n_hidden, n_layers, non_linearity=non_linearity, dropout=dropout)

        self.decoder = nn.Linear(n_hidden, n_tokens)

        self.init_weights()

        self.rnn_type = rnn_type
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.filepath = str(MODELS_DIR / self.__repr__())

    def __repr__(self):
        return (
            f'ChordModel('
            f'RNN_type={self.rnn_type},'
            f'num_layers={self.n_layers},'
            f'num_hidden_layers={self.hidden_layers},'
            f'dropout_prob={self.dropout})'
        )

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        dropped = self.drop(input)
        output, hidden = self.rnn(dropped, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        out = decoded.mean(dim=0)
        #out = nn.Sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                    weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)
        

    def save(self, training=True):
        if training:
            torch.save(self.state_dict(), self.filepath)
        else:
            torch.save(self.state_dict(), self.filepath + '_end')
        print(f'Model {self.__repr__()} saved')
        print('-' * 50)

    def load(self):
        try:
            self.load_state_dict(torch.load(self.filepath,
                                            map_location=lambda storage,
                                                               location: storage)
                                 )
            print(f'Model {self.__repr__()} loaded')
            return True
        except:
            return False

    def predict(self, input):
        self.eval()

        output, _ = self.forward(input)
        f = nn.Sigmoid()
        decoded = f(output)
        if len(decoded.size()) > 1:
            for i in range(output.size()[0]):
                decoded[i][:] = Tensor([v >= 0.5 for v in decoded[i][:]])
        else:
            decoded = Tensor([v >= 0.5 for v in decoded])
        return decoded
