import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from hmmlearn import hmm
from Dataset import *
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm
import pickle
import sys
import gen_gmmhmm_data as ggh
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from matplotlib import pyplot as plt
from gradient_descent import *
from utils import phi, encoding,Fnorm, MAPE, phi_predict
from Density_WFA import density_wfa
from NN_learn_transition import NN_transition_WFA
class stream_rnade(nn.Module):

    def __init__(self, xd, r, mix_n, device, use_batchnorm = True, init_std = 0.1, double_pre = False, initial_bias = None):
        super().__init__()
        self.xd = xd
        self.device = device
        self.use_batchnorm = use_batchnorm

        self.w = torch.nn.Linear(xd, r, bias=True).requires_grad_(True)

        self.mu_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.mu_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)
        self.sig_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.alpha_out = torch.nn.Linear(r, mix_n, bias=True).requires_grad_(True)

        self.double_pre = double_pre
        self.initial_bias = initial_bias
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)


    def forward(self, prev, current, prediction = False):
        if self.double_pre:
            prev = prev.double().to(device)
            current = current.double().to(device)
        else:
            prev = prev.float().to(device)
            current = current.float().to(device)
        tmp_result = phi(self, current.reshape(1, -1), prev, prediction)
        tmp = prev + self.w(current)
        tmp = torch.sigmoid(tmp)
        return tmp_result, tmp


    def lossfunc(self, prev, current):
        conditional_likelihood, tmp = self(prev, current)
        log_likelihood = torch.mean(conditional_likelihood)
        return -log_likelihood, tmp

def fit(train_x, r, lr = 0.001, verbose = True):
    prev = torch.softmax(torch.rand([1, r]), dim = 1).to(device)
    joint_likelihood = 0.
    results = []
    model = stream_rnade(train_x.shape[1], r=r, mix_n=r, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    for i in range(train_x.shape[0]):

        current_x = train_x[i].to(device)
        optimizer.zero_grad()
        pred, _ = model(prev, current_x, prediction = True)
        pred = pred[0]
        loss, prev = model.lossfunc(prev, current_x)
        prev = torch.sigmoid(prev.detach())
        loss.backward()
        index = i
        # if i >= train_x.shape[0] - 200:
        #     index = i - train_x.shape[0] + 200
        joint_likelihood += -loss.detach().cpu().numpy()
        optimizer.step()

        mape = torch.mean(torch.abs(pred.reshape(-1) - current_x.reshape(-1))/(torch.abs(current_x))).detach().cpu().numpy()
        mse = torch.mean((pred.reshape(-1) - current_x.reshape(-1))**2).detach().cpu().numpy()
        print(index, joint_likelihood, -loss, mape, mse)
        if np.isnan(mape) or np.isinf(mape): continue
        # if loss > 10000: continue
        # model.initial_bias = current_x
        results.append([joint_likelihood, -loss.detach().cpu().numpy(), mape, mse])
    return results

if __name__ == '__main__':
    import os
    from nonstationary_HMM import incremental_HMM
    # X = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'weather', 'NEweather_data.csv'), delimiter=',')
    # X = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'movingSquares', 'movingSquares.data'), delimiter=' ')[:10000]
    # data = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'Elec2', 'elec2_data.dat'),
    #                      skip_header=1,
    #                      skip_footer=1,
    #                      names=True,
    #                      dtype=float,
    #                      delimiter=' ')
    # new_data = []
    # for i in range(len(data)):
    #     x = []
    #     for j in range(len(data[i])):
    #         x.append(data[i][j])
    #     new_data.append(x)
    # new_data = np.asarray(new_data)
    # X = new_data

    nshmm = incremental_HMM(r = 3)
    X = nshmm.sample(1000)
    X = np.asarray(X).swapaxes(0, 1)
    ground_truth_joint, ground_truth_conditionals = nshmm.score(X, stream = True)
    ground_truth_conditionals = np.asarray(ground_truth_conditionals)
    ground_truth_joint = np.asarray(ground_truth_joint)


    X = torch.tensor(X).to(device)

    lr = 0.01

    results = fit(X, r = 3, lr=lr, verbose=True)

    results = np.asarray(results)

    plt.plot(results[:, 0], label='pred')
    plt.plot(ground_truth_joint, label='ground')
    plt.title('joint from begining')
    plt.legend()
    plt.show()

    conditional_diff = results[:, 1] - ground_truth_conditionals.reshape(results[:, 1].shape)

    plt.plot(conditional_diff)
    plt.title('conditional likelihood ratio')
    plt.show()

    for ahead in [X.shape[0]-200]:
        tmp_true = ground_truth_conditionals[ahead:]
        tmp_pred = results[ahead:, 1]
        tmp_joint_true = [tmp_true[0]]
        tmp_joint_pred = [tmp_pred[0]]
        for i in range(1, len(tmp_pred)):
            tmp_joint_pred.append(tmp_pred[i] + tmp_joint_pred[i-1])
            tmp_joint_true.append(tmp_true[i] + tmp_joint_true[i - 1])

        plt.plot(tmp_joint_pred, label='pred')
        plt.plot(tmp_joint_true, label='ground')
        plt.title(f'joint from {ahead} steps ahead')
        plt.legend()
        plt.show()


    plt.plot(results[:, 1], label = 'pred')
    plt.plot(ground_truth_conditionals, label = 'ground')
    plt.title('conditional')
    plt.legend()
    plt.show()
    plt.plot(results[:, 2], label = 'mape')
    plt.title('mape')
    plt.show()
    plt.plot(results[:, 3], label = 'mse')
    plt.title('mse')
    plt.show()
    plt.plot(results[:, 4], label='std')
    plt.title('std')
    plt.show()
