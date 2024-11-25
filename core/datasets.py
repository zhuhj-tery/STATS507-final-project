import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.sampler import BatchSampler, Sampler
import json
import bisect


class LoadOfflineFiles(object):

    def __init__(self, yaml_conf, root=r'./datasets'):

        if "metric" in yaml_conf['dataset']:
            self.input_dim, self.output_dim = 28, 1
            print(f"DATA PATH {os.path.join(yaml_conf['dataset'] + r'/state_train.npy')}")
            state_array_train = np.load(os.path.join(yaml_conf['dataset'] + r'/state_train.npy'))
            value_array_train = np.load(os.path.join(yaml_conf['dataset'] + r'/value_train.npy')).reshape((-1,1))
            state_array_test = np.load(os.path.join(yaml_conf['dataset'] + r'/state_test.npy'))
            value_array_test = np.load(os.path.join(yaml_conf['dataset'] + r'/value_test.npy')).reshape((-1,1))
            print(state_array_train.shape, value_array_train.shape, state_array_test.shape, value_array_test.shape)
            state_tensor_train = torch.tensor(state_array_train, dtype=torch.float32)
            value_tensor_train = torch.tensor(value_array_train, dtype=torch.float32)
            state_tensor_test = torch.tensor(state_array_test, dtype=torch.float32)
            value_tensor_test = torch.tensor(value_array_test, dtype=torch.float32)
            self.x_tr, self.y_tr = state_tensor_train, value_tensor_train
            self.x_val, self.y_val = state_tensor_test, value_tensor_test

        else:
            raise NotImplementedError(
                'Wrong dataset name %s (choose one from [mnist, news20])' % yaml_conf['dataset'])


class LoadOfflineFilesSplitPosNeg(object):
    def __init__(self, yaml_conf, root=r'./datasets'):

        if "metric" in yaml_conf['dataset']:
            self.input_dim, self.output_dim = 28, 1
            print(f"DATA PATH {os.path.join(yaml_conf['dataset'] + r'/state_train.npy')}")
            state_array_train_pos = np.load(os.path.join(yaml_conf['dataset'] + r'/positive/state_train.npy'))
            value_array_train_pos = np.load(os.path.join(yaml_conf['dataset'] + r'/positive/value_train.npy')).reshape((-1,1))
            state_array_test_pos = np.load(os.path.join(yaml_conf['dataset'] + r'/positive/state_test.npy'))
            value_array_test_pos = np.load(os.path.join(yaml_conf['dataset'] + r'/positive/value_test.npy')).reshape((-1,1))
            print("Positive data shape:", state_array_train_pos.shape, value_array_train_pos.shape, state_array_test_pos.shape, value_array_test_pos.shape)
            state_tensor_train_pos = torch.tensor(state_array_train_pos, dtype=torch.float32)
            value_tensor_train_pos = torch.tensor(value_array_train_pos, dtype=torch.float32)
            state_tensor_test_pos = torch.tensor(state_array_test_pos, dtype=torch.float32)
            value_tensor_test_pos = torch.tensor(value_array_test_pos, dtype=torch.float32)
            
            state_array_train_neg = np.load(os.path.join(yaml_conf['dataset'] + r'/negative/state_train.npy'))
            value_array_train_neg = np.load(os.path.join(yaml_conf['dataset'] + r'/negative/value_train.npy')).reshape((-1,1))
            state_array_test_neg = np.load(os.path.join(yaml_conf['dataset'] + r'/negative/state_test.npy'))
            value_array_test_neg = np.load(os.path.join(yaml_conf['dataset'] + r'/negative/value_test.npy')).reshape((-1,1))
            print("Negative data shape:", state_array_train_neg.shape, value_array_train_neg.shape, state_array_test_neg.shape, value_array_test_neg.shape)
            state_tensor_train_neg = torch.tensor(state_array_train_neg, dtype=torch.float32)
            value_tensor_train_neg = torch.tensor(value_array_train_neg, dtype=torch.float32)
            state_tensor_test_neg = torch.tensor(state_array_test_neg, dtype=torch.float32)
            value_tensor_test_neg = torch.tensor(value_array_test_neg, dtype=torch.float32)

            self.x_tr_pos, self.y_tr_pos = state_tensor_train_pos, value_tensor_train_pos
            self.x_val_pos, self.y_val_pos = state_tensor_test_pos, value_tensor_test_pos
            self.x_tr_neg, self.y_tr_neg = state_tensor_train_neg, value_tensor_train_neg
            self.x_val_neg, self.y_val_neg = state_tensor_test_neg, value_tensor_test_neg

        else:
            raise NotImplementedError(
                'Wrong dataset name %s (choose one from [mnist, news20])' % yaml_conf['dataset'])



class DataController(Dataset):

    def __init__(self, data, is_train=True):

        if is_train:
            self.x, self.y = data.x_tr, data.y_tr
        else:
            self.x, self.y = data.x_val, data.y_val

        self.input_dim = data.input_dim
        self.output_dim = data.output_dim

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}
    

class NeuralMetricDataController(Dataset):
    """
    This class is used to load the data for neural metric learning, the input data is splitted into two parts, one is the positive data, the other is the negative data. Each time, the data loader will randomly sample a positive data and a negative data.
    """

    def __init__(self, data, is_train=True):

        if is_train:
            self.pos_x, self.pos_y = data.x_tr_pos, data.y_tr_pos
            self.neg_x, self.neg_y = data.x_tr_neg, data.y_tr_neg
        else:
            self.pos_x, self.pos_y = data.x_val_pos, data.y_val_pos
            self.neg_x, self.neg_y = data.x_val_neg, data.y_val_neg

        self.input_dim = data.input_dim
        self.output_dim = data.output_dim

    def __len__(self):
        return 2 * len(self.pos_x)

    def __getitem__(self, idx):
        # randomly sample a positive data and a negative data
        pos_idx = np.random.randint(0, len(self.pos_x))
        neg_idx = np.random.randint(0, len(self.neg_x))
        return {'pos_x': self.pos_x[pos_idx], 'pos_y': self.pos_y[pos_idx], 'neg_x': self.neg_x[neg_idx], 'neg_y': self.neg_y[neg_idx]}




def get_loaders(yaml_conf):
    if 'load_dataset_method' in yaml_conf and yaml_conf['load_dataset_method'] == 'split_pos_neg':
        data = LoadOfflineFilesSplitPosNeg(yaml_conf)
        datasets = {'train': NeuralMetricDataController(data, is_train=True),
                    'val': NeuralMetricDataController(data, is_train=False)}
        dataloaders = {x: DataLoader(datasets[x], batch_size=yaml_conf["batch_size"], num_workers=yaml_conf["num_workers"], shuffle=True, collate_fn=collate_fn) for x in ['train', 'val']}

    else:
        data = LoadOfflineFiles(yaml_conf)
        datasets = {'train': DataController(data, is_train=True),
                    'val': DataController(data, is_train=False)}
    
        dataloaders = {x: DataLoader(datasets[x], batch_size=yaml_conf["batch_size"], num_workers=yaml_conf["num_workers"], shuffle=True) for x in ['train', 'val']}

    return dataloaders

def collate_fn(batch):
    pos_x = torch.stack([item['pos_x'] for item in batch])
    pos_y = torch.stack([item['pos_y'] for item in batch])
    neg_x = torch.stack([item['neg_x'] for item in batch])
    neg_y = torch.stack([item['neg_y'] for item in batch])
    x = torch.cat([pos_x, neg_x], axis=0)
    y = torch.cat([pos_y, neg_y], axis=0)
    return {'x': x, 'y': y}


