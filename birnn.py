import numpy as np
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

class BiRNN(nn.Module):
    def __init__(self, v_size, emb_dim_v, hidden_dim):
        super(BiRNN, self).__init__()
        self.v_size = v_size
        self.emb_dim_v = emb_dim_v
        self.hidden_dim = hidden_dim
        # self.rnn_short = nn.RNN(self.emb_dim_v, self.hidden_dim)
        self.rnn_long = nn.GRU(self.emb_dim_v, self.hidden_dim)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v, padding_idx=-2)
        self.decoder = nn.Linear(self.hidden_dim, self.v_size)

    def forward(self, vids_long, vids_short_al, tids, vids_next, tids_next, uids):
        print vids_long
        # raw_input()
        vids_embeddings = self.embedder_v(vids_long)
        hidden_long = self.init_hidden()
        hiddens_long, hidden_long = self.rnn_long(vids_embeddings, hidden_long)
        vids_predicted = F.softmax(self.decoder(hiddens_long))
        return vids_predicted

def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_dim))