import torch
from torch import nn
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
import numpy as np
from hmmlearn import hmm
import pickle
def encoding(model, X):
    # X = model.encoder_1(X)
    # X = torch.relu(X)
    # # X = model.encoder_2(X)
    # return torch.tanh(X)
    return X

def Fnorm(h):
    norm =  torch.norm((h -1), p=2)
    norm = torch.abs(norm)
    return norm



def phi(model, X, h):
    # print(h.shape)
    # h = torch.tanh(h)
    # print('h1', h)
    # for i in range(len(model.nade_layers)):
    #     h = model.nade_layers[i](h)
    #     h = torch.tanh(h)
    # print(h.shape)
    mu = model.mu_out(h)
    sig = torch.exp(model.sig_out(h))
    # print(torch.mean(sig))
    # print(h)
    # h = h - torch.max(h)
    # print('h', h)
    # print('alpha', model.alpha_out(h))
    # print(h.shape)
    tmp = torch.tanh(model.alpha_out(h))
    alpha = torch.softmax(tmp, dim=1)
    return torch_mixture_gaussian(X, mu, sig, alpha)


def torch_mixture_gaussian(X, mu, sig, alpha):
    mix = D.Categorical(alpha) 
    comp = D.Normal(mu, sig)
    gmm = mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm.log_prob(X)


def gen_hmm_parameters(r = 3):
    init = np.random.rand(r)
    init = init/np.sum(init)
    transition = np.random.rand(r, r)
    # print(transition.shape)
    for i in range(r):
        transition[i, :] /= np.sum(transition[i, :])
    mu = np.random.rand(r) * 2 -1
    mu = mu.reshape(r, 1)
    # print(mu.shape)
    var = np.tile(np.identity(1), (r, 1, 1))
    return init, transition, mu, var

if __name__ == '__main__':
    rs = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
    for r in rs:
        hmmmodel = hmm.GaussianHMM(n_components=r, covariance_type="full")
        hmmmodel.startprob_, hmmmodel.transmat_, hmmmodel.means_, hmmmodel.covars_ = gen_hmm_parameters(r)
        with open('rank_'+str(r), 'wb') as f:
            pickle.dump(hmmmodel, f)


