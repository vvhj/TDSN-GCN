# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
edge = ((4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14))
# visualization
import time

# operation
from . import tools
def random_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data_numpy[:, random_list]
def random_choose_simple(data_numpy, size, center=False):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def uniform_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[:, uniform_list]


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 final_size=-1,
                 debug=False,
                 mmap=True,
                 center_choose=False,
                 bone=False,
                 inchannel=3):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.final_size=final_size
        self.bone = bone
        self.center_choose = center_choose
        self.inchannel = inchannel

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')[:, :self.inchannel]
        else:
            self.data = np.load(self.data_path)[:, :self.inchannel]
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM

        C, T, V, M = data_numpy.shape
        while(T==0):
            index = int(torch.randint(0, len(self.data), (1,)))
            data_numpy = self.data[index]
            label = int(self.label[index])
            sample_name = self.sample_name[index]
            data_numpy = np.array(data_numpy)
            data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM
            C, T, V, M = data_numpy.shape
        # data transform

        # data_numpy = pad_recurrent_fix(data_numpy, self.window_size)  # if short: pad recurrent
        # data_numpy = uniform_sample_np(data_numpy, self.window_size)  # if long: resize
        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
            # data_numpy = random_choose_simple(data_numpy, self.final_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        if self.center_choose:
            # data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)

        return data_numpy.astype(np.float32), label, index