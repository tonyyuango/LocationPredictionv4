import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import DataSet
from birnn import BiRNN

class ModelManager:
    def __init__(self, opt):
        self.opt = opt

    def build_model(self, model_type, dataset):
        u_size = dataset.u_vocab.size()
        v_size = dataset.v_vocab.size()
        t_size = dataset.t_vocab_size
        model = self.init_model(model_type, u_size, v_size, t_size)
        if self.opt['load_model']:
            self.load_model(model, model_type, self.opt['iter'])
            train_time = 0.0
            return model, train_time
        trainer = Trainer(model, self.opt, model_type)
        train_time, best_epoch = trainer.train(dataset.train_loader, dataset.test_loader, self)
        self.load_model(model, model_type, best_epoch)
        return model, train_time

    def load_model(self, model, model_type, epoch):
        # TODO add
        pass

    def init_model(self, model_type, u_size, v_size, t_size):
        if model_type == 'birnn':
            return BiRNN(v_size, self.opt['emb_dim_v'], self.opt['hidden_dim'])

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

    def train(self, train_data, test_data, model_manager):
        best_hr1 = 0
        best_epoch = 0
        start = time.time()
        for epoch in xrange(self.n_epoch):
            self.train_one_epoch(train_data, epoch)
            if (epoch + 1) % self.save_gap == 0:
                valid_hr1 = self.valid_one_epoch(test_data, epoch)
                if valid_hr1 >= best_hr1:
                    best_hr1 = valid_hr1
                    best_epoch = epoch
                    model_manager.save_model(self.model, self.model_type)
        end = time.time()
        return end - start, best_epoch

    def train_one_epoch(self, train_data, epoch):
        total_loss = 0.0
        for i, data_batch in enumerate(train_data):
            self.optimizer.zero_grad()
            session_idx = data_batch[0]
            vids_long = Variable(data_batch[0])
            vids_short_al = Variable(data_batch[1])
            tids = Variable(data_batch[2])
            len_long = Variable(data_batch[3])
            len_short_al = Variable(data_batch[4])
            mask_long = Variable(data_batch[5])
            mask_short_al = Variable(data_batch[6])
            vids_next = Variable(data_batch[7])
            tids_next = Variable(data_batch[8])
            uids = Variable(data_batch[9])
            outputs = self.model(vids_long, len_long)
            loss = self.criterion(outputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]

if __name__ == "__main__":
    torch.manual_seed(7)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationCuda/small/'
    dataset_name = 'foursquare'
    opt = {'u_vocab_file': root_path + dataset_name + '/' + 'u.txt',
           'v_vocab_file': root_path + dataset_name + '/' + 'v.txt',
           'train_data_file': root_path + dataset_name + '/' + 'train.txt',
           'test_data_file': root_path + dataset_name + '/' + 'test.txt',
           'coor_nor_file': root_path + dataset_name + '/' + 'coor_nor.txt',
           'train_log_file': root_path + dataset_name + '/' + 'log.txt',
           'id_offset': 1,
           'n_epoch': 50,
           'batch_size': 2,
           'data_worker': 1,
           'load_model': False,
           'emb_dim_v': 32,
           'hidden_dim': 16,
           'save_gap': 5
           }
    dataset = DataSet(opt)
    manager = ModelManager(opt)
    manager.build_model('birnn', dataset)