import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import format_list_to_string
from torch.autograd import Variable
import numpy as np
use_cuda = torch.cuda.is_available()


class Evaluator:
    def test(self, model, test_data):
        model.eval()
        hits = np.zeros(3)
        cnt = 0
        for data_batch in test_data:
            vids_long, len_long, vids_short_al, len_short_al, short_cnt, mask_long, vids_next, mask_optim, mask_test = self.convert_to_variable(data_batch)
            outputs = self.model(vids_long, len_long, vids_short_al, len_short_al, short_cnt, mask_long, mask_optim, mask_test)
            hits_batch = self.get_hits(outputs, vids_next)
            hits += hits_batch
            cnt += outputs.size(0)
        hits /= cnt
        return hits

    def get_hits(self, outputs, ground_truths):
        print outputs
        print ground_truths
        return np.zeros(3), 0

    def convert_to_variable(self, data_batch):
        vids_long = Variable(data_batch[0])
        vids_short_al = Variable(data_batch[1])
        tids = Variable(data_batch[2])
        len_long = Variable(data_batch[3])
        len_short_al = Variable(data_batch[4])
        mask_long = Variable(data_batch[5])
        mask_optim = Variable(data_batch[6])
        vids_next = Variable(data_batch[7]).masked_select(mask_optim)
        tids_next = Variable(data_batch[8])
        uids = Variable(data_batch[9])
        test_idx = Variable(data_batch[10])
        short_cnt = Variable(data_batch[11])
        print 'test_idx: ', test_idx
        print 'len_long: ', len_long
        mask_al = []
        for uid in xrange(len_long.size(0)):
            for idx in xrange(len_long.data[uid, 0]):
                if idx < test_idx.data[uid, 0]:
                    mask_al.append(0)
                else:
                    mask_al.append(1)
        mask_test = Variable(torch.LongTensor(mask_al)).byte()
        print mask_test
        return vids_long, len_long, vids_short_al, len_short_al, short_cnt, mask_long, vids_next, mask_optim, mask_test