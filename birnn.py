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
        # self.rnn_short = nn.RNN(self.emb_dim_v, self.hidden_dim, batch_first=True)
        self.rnn = nn.GRU(self.emb_dim_v, self.hidden_dim, batch_first=True)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v, padding_idx=0)
        self.decoder = nn.Linear(self.hidden_dim, self.v_size)

    # def forward(self, vids_short_al, len_short_al, short_cnt):
    #     batch_size = short_cnt.data[0,0];
    #     vids_embeddings = self.embedder_v(vids_short_al[0]).index_select(0, Variable(torch.LongTensor([i for i in xrange(batch_size)])))
    #     print vids_embeddings
    #     hidden_short = self.init_hidden(short_cnt.data[0,0])
    #     hiddens_short, hidden_short = self.rnn(vids_embeddings, hidden_short)
    #     print 'hiddens_short: ', hiddens_short
    #     raw_input()
    #     # print hiddens_long
    #     # scores = F.softmax(self.decoder(hiddens_long[0, 3].unsqueeze(0)))
    #     # print scores
    #     raw_input()

    # def forward(self, vids_short_al, len_short_al, short_cnt):
    #     session_cnt_total = torch.sum(short_cnt, dim=0).data[0, 0]
    #     max_session_cnt_a_user = torch.max(short_cnt).data[0]
    #     max_session_length_a_user = torch.max(len_short_al).data[0]
    #     user_cnt = vids_short_al.size(0)
    #     print 'u_cnt: ', user_cnt
    #     print 'max_session_cnt_a_user: ', max_session_cnt_a_user
    #     print 'max_session_length_a_user: ', max_session_length_a_user
    #     print short_cnt.data.numpy()
    #     print len_short_al.data.numpy()
    #     idx_al = []
    #     idx = 0
    #     for u in xrange(user_cnt):
    #         for session_cnt in xrange(max_session_cnt_a_user):
    #             for i in xrange(max_session_length_a_user):
    #                 if i < len_short_al.data[u, session_cnt]:
    #                     idx_al.append(idx)
    #                 idx += 1
    #     print idx_al
    #
    #     # raw_input()
    #     seq_size = vids_short_al.size(2)
    #     vids_short_al_linear = vids_short_al.view(-1, seq_size)
    #     len_short_al_linear = len_short_al.view(-1)
    #     len_short_sorted, idx_sorted = len_short_al_linear.topk(k=session_cnt_total, largest=True, sorted=True)
    #     vids_short_al_valid = vids_short_al_linear.index_select(0, idx_sorted)
    #     vids_embeddings_sorted = self.embedder_v(vids_short_al_valid)
    #     vids_embeddings_sorted_packed = pack_padded_sequence(vids_embeddings_sorted, len_short_sorted.data.numpy(),
    #                                                          batch_first=True)
    #     hidden_short = self.init_hidden(session_cnt_total)
    #     hiddens_short, hidden_short = self.rnn(vids_embeddings_sorted_packed, hidden_short)
    #     unpack_hiddens, unpacked_len = pad_packed_sequence(hiddens_short, batch_first=True)
    #     _, idx_original = idx_sorted.sort(0, descending=False)
    #     index_original_idx = idx_original.view(-1, 1, 1).expand_as(unpack_hiddens)
    #     output = unpack_hiddens.gather(0, index_original_idx.long())
    #     print output.view(-1, self.hidden_dim)
    #     print output.view(-1, self.hidden_dim).index_select(0, Variable(torch.LongTensor(idx_al)))
    #     # print output.view(user_cnt, -1, self.hidden_dim)
    #     raw_input()


    def forward(self, vids_long, len_long):
        len_long_sorted, idx_sorted = len_long.sort(0, descending=True)
        index_sorted_idx = idx_sorted.view(-1, 1).expand_as(vids_long)
        vids_long_sorted = vids_long.gather(0, index_sorted_idx.long())
        vids_embeddings_sorted = self.embedder_v(vids_long_sorted)
        vids_embeddings_sorted_packed = pack_padded_sequence(vids_embeddings_sorted, len_long_sorted.data.numpy(), batch_first=True)
        hidden_long = self.init_hidden(vids_long.size(0))
        hiddens_long, hidden_long = self.rnn(vids_embeddings_sorted_packed, hidden_long)
        unpack_hiddens, unpacked_len = pad_packed_sequence(hiddens_long, batch_first=True)
        _, idx_original = idx_sorted.sort(0, descending=False)
        index_original_idx = idx_original.view(-1, 1, 1).expand_as(unpack_hiddens)
        output = unpack_hiddens.gather(0, index_original_idx.long())
        print output
        raw_input()

    # def forward(self, vids_long, len_long):
    #     print vids_long
    #     vids_embeddings = self.embedder_v(vids_long)
    #     print vids_embeddings[0]
    #     hidden_long = self.init_hidden(vids_long.size(0))
    #     hiddens_long, hidden_long = self.rnn(vids_embeddings, hidden_long)
    #     # print hiddens_long
    #     print hiddens_long[0]
    #     print hiddens_long[1]
    #     # scores = F.softmax(self.decoder(hiddens_long[0, 3].unsqueeze(0)))
    #     # print scores
    #     raw_input()


    def init_hidden(self, batch_size=1):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return hidden