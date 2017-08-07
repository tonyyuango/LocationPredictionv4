import torch
from utils import format_list_to_string
from birnn import BiRNN
from trainer import Trainer

class ModelManager:
    def __init__(self, opt):
        self.opt = opt
        self.model_path = opt['path'] + 'model/'

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

    def init_model(self, model_type, u_size, v_size, t_size):
        if model_type == 'birnn':
            return BiRNN(v_size, self.opt['emb_dim_v'], self.opt['hidden_dim'])


    def get_model_name(self, model_type, epoch):
        emb_dim_v = self.opt['emb_dim_v']
        hidden_dim = self.opt['hidden_dim']
        batch_size = self.opt['batch_size']
        n_epoch = self.opt['n_epoch']
        dp = self.opt['dropout']
        attributes = [model_type, 'DH', hidden_dim, 'DV', emb_dim_v, 'B', batch_size, 'NE', n_epoch, 'E', epoch,
                      'dp', dp]
        model_name = format_list_to_string(attributes, '_')
        return model_name + '.model'

    def load_model(self, model, model_type):
        model_name = self.get_model_name(model_type)
        file_name = self.model_path + model_name
        model.load_state_dict(torch.load(file_name))

    def save_model(self, model, model_type, epoch):
        model_name = self.get_model_name(model_type, epoch)
        file_name = self.model_path + model_name
        torch.save(model.state_dict(), file_name)
