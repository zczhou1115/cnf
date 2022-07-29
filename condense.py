import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os,pickle,json,sys,shutil,math
import numpy as np
import copy
from datetime import datetime
sys.path.append('/home/anson/undergraky/sumvacation')
from tools import setup_seed,makedir

basedir = '/home/anson/undergraky/sumvacation/main'


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

class traindata(Dataset):
    def __init__(self,data_feature,data_target):
        self.len = len(data_feature)
        self.feature = data_feature
        self.target = data_target

    def __getitem__(self,index):
        return self.feature[index],self.target[index]

    def __len__(self):
        return self.len

def get_model(layer_num,neu_num,act_name,inputdim,outputdim):
    if layer_num == 2:
        neunumlist = [inputdim,neu_num,outputdim]
        actlist = ['%s'%act_name]
    if layer_num == 3:
        neunumlist = [inputdim,neu_num,neu_num,outputdim]
        actlist = ['%s'%act_name,'%s'%act_name]
    if layer_num == 4:
        neunumlist = [inputdim,neu_num,neu_num,neu_num,outputdim]
        actlist = ['%s'%act_name,'%s'%act_name,'%s'%act_name]
    if layer_num == 5:
        neunumlist = [inputdim,neu_num,neu_num,neu_num,neu_num,outputdim]
        actlist = ['%s'%act_name,'%s'%act_name,'%s'%act_name,'%s'%act_name]
    return neunumlist,actlist

def init_xavier_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def weight_normal_init(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data,0,0.005)
        nn.init.normal_(m.bias.data,0,0.005)

def gain_model_whole_grad(model):
    grads = []
    for p in model.parameters():
        # print(p.grad.cpu().detach().numpy().flatten())
        grads.append(np.linalg.norm(p.grad.cpu().detach().numpy().flatten()))
    # print(grads)
    grads_length = sum([x**2 for x in grads])
    return math.sqrt(grads_length)

def gain_model_weight_vector(model,layer_num):
    vector = []
    weight = []
    bias = []
    for key,value in model.state_dict().items():
        if 'weight' in key:
            weight.append(copy.deepcopy(value.cpu().detach().numpy()))
        if 'bias' in key:
            tmp = torch.unsqueeze(copy.deepcopy(value),1)
            bias.append(tmp.cpu().detach().numpy())
    # print(bias[0].shape)
    for i in range(layer_num):
        # print(weight[i].shape)
        # print(bias[i].shape)
        tmpvector = np.hstack((weight[i],bias[i]))
        # print(tmpvector.shape)
        vector.append(tmpvector)
    # print(vector[0].reshape)
    return vector
hyperpara = {}
hyperpara['seed'] = 1023
hyperpara['learning_rate'] = 2e-3
hyperpara['x_start'] = -torch.pi
hyperpara['x_end'] = torch.pi
hyperpara['samples'] = 80
hyperpara['inputdim'] = 5
hyperpara['outputdim'] = 1
hyperpara['layer_num'] = 2
hyperpara['neunum'] = 50
hyperpara['act_name'] = 'tanh'
hyperpara['batch_size'] = 51
hyperpara['init_xavier_uniform'] = 1
hyperpara['snormal'] = 0
hyperpara['loss_fn'] = 'MSE'
hyperpara['optimizer'] = 'SGD'

# set random seed
setup_seed(seed=hyperpara['seed'])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # generate dataset
# x = torch.rand(hyperpara['samples'],hyperpara['inputdim']) * 6 - 4
# # x = torch.linspace(hyperpara['x_start'], hyperpara['x_end'],hyperpara['samples'])
# # x = torch.unsqueeze(x,-1)
# x = x.to(device)
# # fun = lambda x :torch.sin(x/6) + torch.sin(x)/4
# fun = lambda x: 3.5*torch.sum(torch.sin(5*x),1) + 1
# y = fun(x)
# y = torch.unsqueeze(y,1)
# print(y.shape)

# generate 1d data
x = torch.linspace(hyperpara['x_start'], hyperpara['x_end'],hyperpara['samples'])
train_set = traindata(x,y)
train_set_iter = DataLoader(dataset=train_set,batch_size=hyperpara['batch_size'],shuffle=False,drop_last=True)

hyperpara['neunumlist'],hyperpara['actlist'] = get_model(hyperpara['layer_num'],hyperpara['neunum'],hyperpara['act_name'],hyperpara['inputdim'],hyperpara['outputdim'])
model = FNNnet(neunumlist=hyperpara['neunumlist'],actlist=hyperpara['actlist'])
if hyperpara['init_xavier_uniform'] == 1:
    model.apply(init_xavier_weights)
else:
    pass
if hyperpara['snormal'] == 1:
    model.apply(weight_normal_init)
else:
    pass
model = model.to(device)

# gain_model_whole_grad(model)
# copy experiment code to the save Folder
subFolderName = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
savedir = makedir(os.path.join(basedir,subFolderName))
shutil.copy(__file__,savedir)
# save hyperparameters
para_file = open("%s/para.txt"%(savedir),"w")
para_str = json.dumps(hyperpara,indent=0)
para_file.write(para_str)
para_file.close()
R = {}
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = hyperpara['learning_rate'])
# def train_one_epoch(epoch_index):
#     lossa = 0
#     for i,(x,y) in enumerate(train_set_iter):

#         optimizer.zero_grad()
#         loss = loss_fn(model(x),y)
#         loss.backward()
#         optimizer.step()
#         lossa +=loss.item()
#     return lossa,loss.grad()
# R['losses'] = []
# for epoch in range(1):
#     model.train(True)
#     if epoch % 100 == 0:
#         torch.save(model.state_dict(),os.path.join(savedir,'epoch=%s.pt'%(epoch+1)))
#     avg_loss = train_one_epoch(epoch)
#     R['losses'].append(copy.deepcopy(avg_loss))
R['losses'] = []
R['whole_grad'] = []
R['weightvector'] = []
for epoch in range(10000):
    model.train(True)
    if epoch % 100 == 0:
        torch.save(model.state_dict(),os.path.join(savedir,'epoch=%s.pt'%(epoch+1)))
    if epoch % 10 == 0:
        weightvector = gain_model_weight_vector(model,hyperpara['layer_num'])
        R['weightvector'].append(weightvector)
    optimizer.zero_grad()
    loss = loss_fn(model(x),y)
    R['losses'].append(loss.item())
    loss.backward()
    grad_length = gain_model_whole_grad(model)
    R['whole_grad'].append(grad_length)
    optimizer.step()
torch.save(model.state_dict(),os.path.join(savedir,'epoch=-1.pt'))

with open("%s/para.pkl"%(savedir),'wb') as f0:
    pickle.dump(hyperpara,f0)
    f0.close()

with open(os.path.join(savedir,'trainpro.pkl'),'wb') as f:
    pickle.dump(R,f)
    f.close()    
plt.plot(R['losses'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(savedir,'loss.png'))
plt.close()

# plt.plot(x.cpu().detach().numpy(),model(x).cpu().detach().numpy(),label='train')
# plt.plot(x.cpu().detach().numpy(),fun(x).cpu().detach().numpy(),label='true')
# plt.legend()
# plt.savefig(os.path.join(savedir,'result.png'))
# plt.close()