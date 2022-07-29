from tools import loaddata,makedir
import os,sys,torch
import matplotlib.pyplot as plt
import tools
import torch.nn as nn
import numpy as np

basedir = '/home/anson/undergraky/sumvacation/main'
savedir = makedir(os.path.join(basedir, 'figure'))

data = loaddata('/home/anson/undergraky/sumvacation/main/220729135212/trainpro.pkl')
l1 = data['weightvector'][10][0]
print(l1)
l1normal = np.asarray([x/np.linalg.norm(x) for x in l1])
# print(l1normal)
# inner_matrix = np.dot(l1normal,l1normal.T)
rankvector = np.dot(l1normal,l1normal[0])
index = np.argsort(rankvector)
print(index)
rel1normal = l1normal[index]
inner_matrix = np.dot(rel1normal,rel1normal.T)
# print(inner_matrix.shape)

plt.pcolormesh(inner_matrix,vmin=-1,vmax=1)
plt.colorbar()
plt.show()