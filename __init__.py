import os
import torch
from dataset import DataSet
from model_manager import ModelManager


if __name__ == "__main__":
    torch.manual_seed(3)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationCuda/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationCuda/') \
        else '/shared/data/qyuan/LocationCuda/'
    dataset_name = 'foursquare'
    path = root_path + 'small/' + dataset_name + '/'
    opt = {'path': path,
            'u_vocab_file': path+ 'u.txt',
           'v_vocab_file': path + 'v.txt',
           'train_data_file': path + 'train.txt',
           'test_data_file': path + 'test.txt',
           'coor_nor_file': path + 'coor_nor.txt',
           'train_log_file': path + 'log.txt',
           'id_offset': 1,
           'n_epoch': 2000,
           'batch_size': 50,
           'data_worker': 1,
           'load_model': False,
           'emb_dim_v': 32,
           'hidden_dim': 16,
           'save_gap': 5,
           'dropout': 0.5
           }
    dataset = DataSet(opt)
    manager = ModelManager(opt)
    manager.build_model('birnn', dataset)