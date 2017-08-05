import numpy as np
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class BiRNN(nn.Module):
    def __init__(self, v_size, emb_dim_v, hidden_dim):
        super(BiRNN, self).__init__()
        self.v_size = v_size
        self.emb_dim_v = emb_dim_v
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(self.emb_dim_v, self.hidden_dim, batch_first=True)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v, padding_idx=0)
        self.decoder = nn.Linear(self.hidden_dim, self.v_size)

    def forward(self, vids_long, len_long):
        print 'len_long: ', len_long
        print vids_long
        vids_embeddings = self.embedder_v(vids_long)
        print vids_embeddings
        packed = pack_padded_sequence(vids_embeddings, [3, 2], batch_first=True)
        print 'packed: ', packed
        # print 'unpacked: ', pad_packed_sequence(packed, batch_first=True)
        hidden_long = self.init_hidden(5)
        hiddens_long, hidden_long = self.rnn(packed, hidden_long)
        print 'hiddens_long: ', hiddens_long
        unpack_hiddens, unpacked_len = pad_packed_sequence(hiddens_long, batch_first=True)
        print 'unpack_hiddens: ', unpack_hiddens
        vids_predicted = F.softmax(self.decoder(unpack_hiddens))
        raw_input()
        return vids_predicted

    def init_hidden(self, batch_size=1):
        return Variable(torch.zeros(1, 2, self.hidden_dim))