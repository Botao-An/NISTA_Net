import os
import numpy as np
import joblib
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import progressbar

# activation functions
class ReLU(nn.Module):
    def __init__(self, chan_num, init_lambdas=1e-3):
        super(ReLU, self).__init__()
        self.lambdas = nn.Parameter(init_lambdas * torch.ones(1, chan_num, 1))
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu (x + self.lambdas)
        return out

class Soft(nn.Module):
    def __init__(self, chan_num, init_lambdas=1e-3):
        super(Soft, self).__init__()
        self.lambdas = nn.Parameter(init_lambdas * torch.ones(1,chan_num,1))

    def forward(self, x):
        mask1 = (x > self.lambdas).float()
        mask2 = (x < -self.lambdas).float()
        out = mask1.float() * (x - self.lambdas)
        out += mask2.float() * (x + self.lambdas)
        return out

class Hard(nn.Module):
    def __init__(self, chan_num, init_lambdas=1e-3):
        super(Hard, self).__init__()
        self.lambdas = nn.Parameter(init_lambdas * torch.ones(1,chan_num,1))

    def forward(self, x):
        mask1 = (x > self.lambdas).float()
        mask2 = (x < -self.lambdas).float()
        out = mask1.float() * x + mask2.float() * x
        return out

class Firm(nn.Module):
    def __init__(self, channel_num, init_lam=1e-3, init_mu=2e-3):
        super(Firm, self).__init__()
        self.lam = nn.Parameter(init_lam * torch.ones(1,channel_num,1),requires_grad=True)
        self.mu  = nn.Parameter(init_mu  * torch.ones(1,channel_num,1),requires_grad=True)

    def forward(self, x):
        mask1 = (x >  self.mu)
        mask2 = (x < -self.mu)
        mask3 = (x <=  self.mu) & (x >  self.lam)
        mask4 = (x >= -self.mu) & (x < -self.lam)

        out = mask1.float()*x + mask2.float()*x + \
              mask3.float()*(self.mu/(self.mu-self.lam))*(x-self.lam) + mask4.float()*(self.mu/(self.mu-self.lam))*(self.lam+x)
        return out



# model
class NISTA_Net(nn.Module):
    def __init__(self, opt):
        super(NISTA_Net, self).__init__()
        self.opt = opt
        self.opt.Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
        self.T = opt.unfoldings

        # Initialization
        self.W1 = nn.Parameter(torch.randn(opt.CL[1], opt.CL[0], opt.KL[0]), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(opt.CL[2], opt.CL[1], opt.KL[1]), requires_grad=True)
        self.W3 = nn.Parameter(torch.randn(opt.CL[3], opt.CL[2], opt.KL[2]), requires_grad=True)
        self.W1.data = .1 / np.sqrt(opt.CL[0] * opt.KL[0]) * self.W1.data
        self.W2.data = .1 / np.sqrt(opt.CL[1] * opt.KL[1]) * self.W2.data
        self.W3.data = .1 / np.sqrt(opt.CL[2] * opt.KL[2]) * self.W3.data

        # Optimization Steps
        self.mu1 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.mu2 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.mu3 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

        # Bias in model
        self.b1 = nn.Parameter(torch.zeros(1, opt.CL[1], 1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, opt.CL[2], 1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1, opt.CL[3], 1), requires_grad=True)

        # Threshold Functions
        active_dict  ={'ReLU':ReLU, 'Soft':Soft, 'Firm':Firm}
        self.thrd1 = active_dict[opt.active_type](opt.CL[1])
        self.thrd2 = active_dict[opt.active_type](opt.CL[2])
        self.thrd3 = active_dict[opt.active_type](opt.CL[3])

        # Classifier
        self.Wclass = nn.Linear(opt.CL[3], opt.n_class)



    def forward(self, x, all_out=False):
        # Encoding
        gamma1 = self.thrd1(self.mu1*F.conv1d(x,     self.W1, stride = self.opt.SL[0], padding=self.opt.PL[0]) +self.b1)
        gamma2 = self.thrd2(self.mu2*F.conv1d(gamma1,self.W2, stride = self.opt.SL[1], padding=self.opt.PL[1]) +self.b2)
        gamma3 = self.thrd3(self.mu3*F.conv1d(gamma2,self.W3, stride = self.opt.SL[2], padding=self.opt.PL[2]) +self.b3)

        for _ in  range(self.opt.unfoldings):
            if self.opt.direct_connect != True:
                # backward computation
                gamma2 = F.conv_transpose1d(gamma3,self.W3, stride = self.opt.SL[2], padding=self.opt.PL[2])
                gamma1 = F.conv_transpose1d(gamma2,self.W2, stride = self.opt.SL[1], padding=self.opt.PL[1])
                        
            # forward computation: x(i+1) = relu(x^(i+1)-c*DT*(D*x^(i+1)-x(i)))
            gamma1 = self.thrd1((gamma1 - self.mu1*F.conv1d( F.conv_transpose1d(gamma1,self.W1, stride = self.opt.SL[0], padding=self.opt.PL[0]) - x ,     self.W1, stride = self.opt.SL[0], padding=self.opt.PL[0])) +self.b1)
            gamma2 = self.thrd2((gamma2 - self.mu2*F.conv1d( F.conv_transpose1d(gamma2,self.W2, stride = self.opt.SL[1], padding=self.opt.PL[1]) - gamma1, self.W2, stride = self.opt.SL[1], padding=self.opt.PL[1])) +self.b2)
            gamma3 = self.thrd3((gamma3 - self.mu3*F.conv1d( F.conv_transpose1d(gamma3,self.W3, stride = self.opt.SL[2], padding=self.opt.PL[2]) - gamma2, self.W3, stride = self.opt.SL[2], padding=self.opt.PL[2])) +self.b3)

        # classifier
        if self.opt.pool_type == 'Max':
            gamma_new, index = F.max_pool1d(torch.abs(gamma3), gamma3.shape[2], return_indices=True)
        elif self.opt.pool_type == 'Avg':
            gamma_new        = F.avg_pool1d(torch.abs(gamma3), gamma3.shape[2])
        else:
            raise ValueError("Unexptected pooling type, opt: Max, Avg")

        gamma = gamma_new.view(gamma_new.shape[0], -1)
        out = self.Wclass(gamma)
        out = F.log_softmax(out, dim=1)

        if all_out:

            gamma2 = F.conv_transpose1d(gamma3, self.W3, stride=self.opt.SL[2], padding=self.opt.PL[2])
            gamma1 = F.conv_transpose1d(gamma2, self.W2, stride=self.opt.SL[1], padding=self.opt.PL[1])
            x_hat  = F.conv_transpose1d(gamma1, self.W1, stride=self.opt.SL[0], padding=self.opt.PL[0])

            return out, gamma3, x_hat
        else:
            return out