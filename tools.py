import os
import math
import numpy as np
import torch
import random
import pickle
import torch.nn as nn
class FNNnet(nn.Module):
    def __init__(self,neunumlist=None,actlist=None):
        super(FNNnet,self).__init__()
        self.neunumlist = neunumlist
        self.actlist = actlist        
        self.sequence = nn.Sequential()
        n = len(self.neunumlist)
        for i in range(n-1):
            self.sequence.add_module('linear%s'%i,nn.Linear(self.neunumlist[i],self.neunumlist[i+1]))
            if i < n-2:
                if self.actlist[i] == 'sigmoid':
                    self.sequence.add_module('act%s'%i,nn.Sigmoid())
                elif self.actlist[i] == 'ReLU':
                    self.sequence.add_module('act%s'%i,nn.ReLU())
                elif self.actlist[i] == 'tanh':
                    self.sequence.add_module('act%s'%i,nn.Tanh())
    def forward(self,x):
        x = self.sequence(x)
        return x
def makedir(dirname):
    if os.path.exists(dirname) == True:
        pass
    else:
        os.mkdir(dirname)
    return dirname

def setup_seed(seed=2047):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def loaddata(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    return data

