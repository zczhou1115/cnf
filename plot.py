from tools import loaddata,makedir
import os,sys,torch
import matplotlib.pyplot as plt
import tools
import torch.nn as nn

basedir = '/home/anson/undergraky/sumvacation/main'
savedir = makedir(os.path.join(basedir, 'figure'))

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135712/trainpro.pkl')
# plt.plot(data['losses'],label='3layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135728/trainpro.pkl')
# plt.plot(data['losses'],label='4layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135743/trainpro.pkl')
# plt.plot(data['losses'],label='5layer')
# plt.title('sigmoid+256+kaiminguniforminit')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig(os.path.join(savedir,'sigmoid256km.png'))
# plt.close()



# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135800/trainpro.pkl')
# plt.plot(data['losses'],label='3layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135815/trainpro.pkl')
# plt.plot(data['losses'],label='4layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135832/trainpro.pkl')
# plt.plot(data['losses'],label='5layer')
# plt.title('sigmoid+512+kaiminguniforminit')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig(os.path.join(savedir,'sigmoid512km.png'))
# plt.close()


# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135850/trainpro.pkl')
# plt.plot(data['losses'],label='3layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135905/trainpro.pkl')
# plt.plot(data['losses'],label='4layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135920/trainpro.pkl')
# plt.plot(data['losses'],label='5layer')
# plt.title('sigmoid+256+xiavieruniforminit')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig(os.path.join(savedir,'sigmoid256xaiver.png'))
# plt.close()

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135936/trainpro.pkl')
# plt.plot(data['losses'],label='3layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708135949/trainpro.pkl')
# plt.plot(data['losses'],label='4layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220708140005/trainpro.pkl')
# plt.plot(data['losses'],label='5layer')
# plt.title('tanh+512+xaiveruniforminit')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig(os.path.join(savedir,'sigmoid512xaiver.png'))
# plt.close()

# data = loaddata('/home/anson/undergraky/sumvacation/main/220712224518/trainpro.pkl')
# plt.plot(data['whole_grad'],label='3layer')
# # plt.plot(data['losses'],label='loss')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220712224604/trainpro.pkl')
# plt.plot(data['whole_grad'],label='4layer')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220712224628/trainpro.pkl')
# plt.plot(data['whole_grad'],label='5layer')
# plt.title('tanh+512+kaiminguniform+grad')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(os.path.join(savedir,'tanh512kaiminggrad.png'))
# plt.show()

# data = loaddata('/home/anson/undergraky/sumvacation/main/220712224424/trainpro.pkl')
# plt.plot(data['losses'],label='256kaiming')

# data = loaddata('/home/anson/undergraky/sumvacation/main/220712223701/trainpro.pkl')
# plt.plot(data['losses'],label='256xavier')


# data = loaddata('/home/anson/undergraky/sumvacation/main/220712224518/trainpro.pkl')
# plt.plot(data['losses'],label='512kaiming')
# data = loaddata('/home/anson/undergraky/sumvacation/main/220712223923/trainpro.pkl')
# plt.plot(data['losses'],label='512xavier')
# plt.title('tanh+3layer')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(os.path.join(savedir,'tanh3layer.png'))
# plt.show()

data = loaddata('/home/anson/undergraky/sumvacation/main/220713140955/trainpro.pkl')
plt.plot(data['losses'],label='3layer')
# plt.plot(data['losses'],label='loss')

data = loaddata('/home/anson/undergraky/sumvacation/main/220713141015/trainpro.pkl')
plt.plot(data['losses'],label='4layer')

data = loaddata('/home/anson/undergraky/sumvacation/main/220713141038/trainpro.pkl')
plt.plot(data['losses'],label='5layer')
plt.title('ReLU+512+xavieruniform')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(savedir,'ReLU512xavier.png'))
plt.show()