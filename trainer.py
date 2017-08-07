import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
# use_cuda = False

class Trainer:
    def __init__(self, model, opt, model_type):
        self.opt = opt
        self.train_log_file = opt['train_log_file']
        self.n_epoch = opt['n_epoch']
        self.batch_size = opt['batch_size']
        self.model_type = model_type
        self.save_gap = opt['save_gap']
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, train_data, model_manager):
        start = time.time()
        for epoch in xrange(self.n_epoch):
            epoch_loss = self.train_one_epoch(train_data, epoch)
            if (epoch + 1) % self.save_gap == 0:
                print 'epoch: ', epoch + 1, '\tloss: ', epoch_loss
                model_manager.save_model(self.model, self.model_type, epoch + 1)
        end = time.time()
        return end - start


    def train_one_epoch(self, train_data, epoch):
        total_loss = 0.0
        for i, data_batch in enumerate(train_data):
            self.optimizer.zero_grad()
            vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, vids_next, mask_optim, mask_evaluate = self.convert_to_variable(data_batch)
            outputs = self.model(vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate)
            loss = self.criterion(outputs, vids_next)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]
        # print 'epoch: ', epoch + 1, '\tloss: ', total_loss
        return total_loss

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
        # print torch.sum(len_long - test_idx, 0)
        # raw_input()
        if torch.sum(len_long - test_idx, 0).data[0] > 0:
            mask_evaluate = mask_optim.clone()
            for uid in xrange(len_long.size(0)):
                for idx in xrange(len_long.data[uid, 0]):
                    if idx < test_idx.data[uid, 0]:
                        mask_evaluate.data[uid, idx] = 0
        vids_next = Variable(data_batch[7]).masked_select(mask_optim if mask_evaluate is None else mask_evaluate)
        # print 'len_short_al: ', len_short_al
        # print 'len_long: ', len_long
        # print 'test_idx: ', test_idx
        # print 'vids_next: ', vids_next
        if use_cuda:
            return vids_long.cuda(), len_long.cuda(), vids_short_al.cuda(), len_short_al.cuda(), tids_next, short_cnt.cuda(), mask_long.cuda(), vids_next.cuda(), mask_optim.cuda(), mask_evaluate.cuda()
        else:
            return vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, vids_next, mask_optim, mask_evaluate