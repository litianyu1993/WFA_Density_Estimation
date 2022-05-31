import numpy as np
import torch
from torch import nn
import gen_gmmhmm_data as ggh
relu = nn.ReLU()
tanh = nn.Tanh()
import math

sig = nn.Sigmoid()
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from gradient_descent import *
from hmmlearn import hmm
from Dataset import *
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from utils import phi, encoding, Fnorm
from scipy.stats import norm

# from density_estimator import Hankel, evaluate_gaussian_hmm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class hankel_density_spec(nn.Module):
    def __init__(self, d, xd, r, hidden_dim = None, mixture_number=10, L=5, init_std=1e-3, device=device, double_pre=True,
                 previous_hd=None, encoder_hid=None, use_softmax_norm = False, initial_bias = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2*r
        self.device = device
        self.core_list = nn.ParameterList()
        self.rank = r
        self.input_dim = d
        self.length = L
        self.mixture_number = mixture_number
        for i in range(L):
            tmp_core = torch.normal(0., init_std, [r, d, r])
            tmp_core = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
            self.core_list.append(tmp_core)

        if previous_hd is None:
            self.batchnrom = nn.BatchNorm1d(r)
            self.mu_out = torch.nn.Linear(self.nade_hid[-1], mixture_number*xd, bias=True)
            self.sig_out = torch.nn.Linear(self.nade_hid[-1], mixture_number*xd, bias=True)
            self.alpha_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)
            self.encoder_1 = torch.nn.Linear(xd, d, bias=True)
            tmp_core = torch.normal(0, init_std, [1, r])
            self.init_w = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        else:
            for i in range(len(previous_hd.core_list)):
                self.core_list[i] = previous_hd.core_list[i]
                self.core_list[i].requires_grad = True
            self.batchnrom = previous_hd.batchnrom
            self.batchnrom.requires_grad = True
            self.init_w = previous_hd.init_w
            self.init_w.requires_grad = True
            self.mu_out = previous_hd.mu_out
            self.mu_out.requires_grad = True
            self.sig_out = previous_hd.mu_out
            self.sig_out.requires_grad = True
            self.alpha_out = previous_hd.alpha_out
            self.alpha_out.weight.requires_grad = True
            self.encoder_1 = previous_hd.encoder_1
            self.encoder_1.weight.requires_grad = True
            self.encoder_1.bias.requires_grad = True

        self.double_pre = double_pre
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)

    def forward(self, X):
        # print(X.shape[2], self.length)
        assert X.shape[2] == self.length, "trajectory length does not fit the network structure"
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)

        result = 0.
        norm = 0.
        for i in range(self.length):
            if i == 0:
                tmp = self.init_w.repeat(X.shape[0], 1)
            else:
                tmp_A = self.core_list[i - 1]
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, tmp_A)
                if self.use_softmax_norm:
                    tmp = self.batchnrom(tmp)
            tmp_result = phi(self, X[:, :, i], torch.softmax(tmp, dim = 1))
            norm += Fnorm(tmp)
            result = result + tmp_result
        return result, norm, tmp


    def fit(self, train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler=None, verbose=True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
            train_likehood.append(train(self, self.device, train_loader, optimizer, X=train_x))
            validation_likelihood.append(validate(self, self.device, validation_loader, X=test_x))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(
                    train_likehood[-1],
                    validation_likelihood[-1]))
            if  epoch > 10 and validation_likelihood[-1] > validation_likelihood[-2]:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                    count += 1
                if count >=20: break
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])
        return train_likehood, validation_likelihood

    def eval_likelihood(self, X):
        log_likelihood, hidden_norm, _ = self(X)
        return torch.mean(log_likelihood)

    def lossfunc(self, X):
        log_likelihood, hidden_norm, _ = self(X)
        log_likelihood = torch.mean(log_likelihood)
        return -log_likelihood

class regression_transition(nn.Module):

    def __init__(self,hd, h2d, h2d1, init_std = 0.01, device = device, double_pre = False, linear_transition = False):
        super().__init__()
        self.hd = hd
        self.h2d = h2d
        self.h2d1 = h2d1
        self.mu_out = h2d1.mu_out
        self.sig_out  = h2d1.sig_out
        self.alpha_out = h2d1.alpha_out

