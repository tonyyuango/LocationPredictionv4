import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import format_list_to_string
from torch.autograd import Variable
import numpy as np
use_cuda = torch.cuda.is_available()


class Evaluator:
    def __init__(self, model, opt, model_type):
        self.model = model
        self.opt = opt

    def eval(self, test_data):
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
        raw_input()
        return np.zeros(3), 0

    def convert_to_variable(self, data_batch):
        vids_long = Variable(data_batch[0])
        vids_short_al = Variable(data_batch[1])
        tids = Variable(data_batch[2])
        len_long = Variable(data_batch[3])
        len_short_al = Variable(data_batch[4])
        mask_long = Variable(data_batch[5])
        mask_optim = Variable(data_batch[6])
        tids_next = Variable(data_batch[8])
        uids = Variable(data_batch[9])
        # print uids
        test_idx = Variable(data_batch[10])
        short_cnt = Variable(data_batch[11])
        mask_evaluate = None
        # if there are records for evaluation
        if torch.sum(len_long - test_idx, 0).data[0, 0] > 0:
            mask_evaluate = mask_optim.clone()
            for uid in xrange(len_long.size(0)):
                for idx in xrange(len_long.data[uid, 0]):
                    if idx < test_idx.data[uid, 0]:
                        mask_evaluate.data[uid, idx] = 0
        vids_next = Variable(data_batch[7]).masked_select(mask_optim if mask_evaluate is None else mask_evaluate)
        if use_cuda:
            return vids_long.cuda(), len_long.cuda(), vids_short_al.cuda(), len_short_al.cuda(), short_cnt.cuda(), mask_long.cuda(), vids_next.cuda(), mask_optim.cuda(), mask_evaluate.cuda()
        else:
            return vids_long, len_long, vids_short_al, len_short_al, short_cnt, mask_long, vids_next, mask_optim, mask_evaluate