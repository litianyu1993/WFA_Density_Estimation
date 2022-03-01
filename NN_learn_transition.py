import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from hmmlearn import hmm
from Dataset import *
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm, insert_bias
import pickle
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from utils import encoding, phi
from gradient_descent import train, validate
from matplotlib import pyplot as plt

class NN_transition_WFA(nn.Module):

    def __init__(self,hd, h2d, h2d1, init_std = 0.1, device = device, double_pre = False, linear_transition = False):
        super().__init__()
        self.hd = hd
        self.h2d = h2d
        self.h2d1 = h2d1
        self.hankel_2l = h2d.core_list
        self.hankel_l = hd.core_list
        self.hankel_2l1 = h2d1.core_list
        # print(self.hd.init_w.shape, self.hankel_l[0].shape, self.hankel_2l[0].shape, self.hankel_2l1[0].shape)
        self.hankel_l[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.hd.init_w, self.hankel_l[0]).squeeze())
        # self.hankel_2l[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.h2d.init_w, self.hankel_2l[0]).squeeze())
        # self.hankel_2l1[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.h2d1.init_w, self.hankel_2l1[0]).squeeze())

        self.l = len(hd.core_list)
        self.d = hd.core_list[0].shape[0]
        self.r = hd.core_list[0].shape[1]

        self.nade_layers = hd.nade_layers
        self.mu_out = hd.mu_out
        self.sig_out = hd.sig_out
        self.alpha_out = hd.alpha_out
        self.encoder_1 = hd.encoder_1
        self.encoder_2 = hd.encoder_2


        self.device = device
        tmp_core =torch.normal(0, init_std, [self.r, self.d, self.r]).float()
        self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))


        self.hankel_2l = self.qr_svd(self.hankel_2l, self.l)
        self.init_w = self.comput_initial_vecotr(self.hankel_l, self.hankel_2l)
        self.double_pre = double_pre
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)

        if linear_transition:
            self.tran_activation = torch.nn.Linear(self.r, self.r, bias=False).to(device)

        else:
            self.tran_activation = nn.BatchNorm1d(self.A.shape[-1])


    def comput_initial_vecotr(self, hl, h2l):
        tmp = []
        for i in range(self.l):
            j = self.l+i
            if i == 0:
                # print(h2l[j].shape, hl[i].shape)
                tmp_core = torch.einsum('ijk, jl -> ikl', h2l[j], hl[i]).reshape(self.r, -1)
            elif i == self.l-1:
                tmp_core = torch.einsum('ijp, qjp -> iq', h2l[j], hl[i]).reshape(self.r*self.r, 1)
            else:
                tmp_core = torch.einsum('ijk, qjp -> iqkp', h2l[j], hl[i]).reshape(self.r**2, self.r**2)
            tmp.append(tmp_core)
        contraction = tmp[0]
        for i in range(1, self.l):
            contraction = contraction @ tmp[i]
        return contraction.squeeze().detach()

    def forward(self, X2l1):
        if self.double_pre:
            X = X2l1.double().to(device)
        else:
            X = X2l1.float().to(device)
        result = 0.

        for i in range(X.shape[2]):
            if i == 0:
                tmp = self.h2d.init_w
            if i == self.l:
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, self.A)
                # tmp = torch.tanh(tmp)
                # print(tmp.shape)
                tmp = self.tran_activation(tmp)
            else:
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, self.hankel_2l[i - 1])

            tmp_result = phi(self, X[:, :, i].squeeze(), tmp)
            result = result + tmp_result
        return result



    def lossfunc(self, X):
        target, _ = self.h2d1(X)
        pred = self(X)
        return torch.mean((target - pred) **2)


    def fit(self, train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler=None, verbose=True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
            train_likehood.append(train(self, self.device, train_loader, optimizer, X=train_x).detach().to('cpu'))
            validation_likelihood.append(validate(self, self.device, validation_loader, X=test_x).detach().to('cpu'))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(
                    train_likehood[-1],
                    validation_likelihood[-1]))
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])

        # self.core_list[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]).squeeze())
        return train_likehood, validation_likelihood

    def qr_svd(self, tensor, l):
        r = self.r
        d = self.d
        singular_values = torch.eye(tensor[0].shape[-1]).to(device)
        for i in range(l):
            # if i == 0:
            #     B, C = torch.linalg.qr(tensor[i].transpose(0, 1), mode = 'complete')
            #     tensor[i].data = C.transpose(0, 1)
            #     tmp = B.transpose(0, 1)
            #     tensor[i+1].data  = torch.einsum('ij, jkl -> ikl', tmp, tensor[i+1])
            # else:
            core = tensor[i].reshape(-1, tensor[i].shape[-1])
            B, C = torch.linalg.qr(core.transpose(0, 1).contiguous(), mode = 'complete')
            tensor[i].data = C.transpose(0, 1).contiguous().reshape([r, d, r])
            tmp = B.transpose(0, 1).contiguous()
            if i != l-1:
                tensor[i+1].data  = torch.einsum('ij, jkl -> ikl', tmp, tensor[i+1])
            else:
                singular_values = singular_values @ B.transpose(0, 1).contiguous()

        for i in range(2*l-1, l-1, -1):
            core  = tensor[i].reshape(r, d*r)
            B, C = torch.linalg.qr(core, mode = 'complete')
            tensor[i].data = C.reshape([r, d, r])
            if i != l:
                tensor[i-1].data = torch.einsum('jkl, li -> jki', tensor[i-1], B)
            else:
                singular_values = singular_values @ B
        tensor[l- 1].data = tensor[l-1] @ singular_values

        return tensor




