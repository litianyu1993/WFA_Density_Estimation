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
class density_wfa(nn.Module):

    def __init__(self, xd, d, r, mix_n, device, use_batchnorm = True, init_std = 1, double_pre = False, initial_bias = None):
        super().__init__()
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.encoder_1 = torch.nn.Linear(xd, d, bias=True)

        tmp_core = torch.normal(0, init_std, [1, r])
        self.init_w = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        tmp_core = torch.normal(0, init_std, [r, d, r])
        self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))

        self.mu_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.sig_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.alpha_out = torch.nn.Linear(r, mix_n, bias=True).requires_grad_(True)

        self.mu_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)
        self.sig_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)
        self.alpha_out2 = torch.nn.Linear(mix_n, mix_n, bias=True).requires_grad_(True)

        self.batchnrom = nn.BatchNorm1d(self.A.shape[-1])
        self.double_pre = double_pre
        self.initial_bias = initial_bias
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)

        self.m = nn.Softsign()

    def forward(self, X, prediction = False):
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)
        # print(X.device)
        result = 0.
        norm = 0.
        for i in range(X.shape[2]):
            if i == 0:
                tmp = self.init_w.repeat(X.shape[0], 1)
            else:
                tran = self.A
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, tran)
            if self.use_batchnorm:
                # tmp = torch.softmax(tmp, dim = 1)
                tmp = self.batchnrom(tmp)
            # print(self.init_w)
            # print(i, 0, tmp, torch.any(torch.isnan(tmp)))
            tmp_result = phi(self, X[:, :, i], torch.softmax(tmp, dim  =1), prediction)
            # print(3, tmp, torch.any(torch.isnan(tmp)))
            if not prediction:
                norm += Fnorm(tmp)
                result = result + tmp_result
        if prediction:
            return tmp_result
        else:
            return result, norm

    def fit(self,train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler = None, verbose = True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
            train_likehood.append(train(self, self.device, train_loader, optimizer, X = train_x, rescale=False))
            validation_likelihood.append(validate(self, self.device, validation_loader, X =test_x))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(train_likehood[-1],
                                                                                                     validation_likelihood[-1]))
            if epoch > 5 and validation_likelihood[-1] > validation_likelihood[-2]:
                count += 1
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                if count > 20: break
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])

        return train_likehood, validation_likelihood

    def eval_likelihood(self, X, batch = False):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = log_likelihood.detach().cpu().numpy()
        if not batch:
            return np.mean(log_likelihood)
        else:
            return log_likelihood

    def lossfunc(self, X):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = torch.mean(log_likelihood)
        return -log_likelihood


    def bootstrapping(self, X):
        tmp_x = X[:, :, :1]
        for i in range(1, X.shape[-1]):
            # print(tmp_x.shape)
            tmp_y = X[:, :, i]
            pred_mu, pred_sig = self(tmp_x, prediction=True)
            pred_mu = pred_mu.reshape(tmp_y.shape)
            tmp_x = torch.cat((tmp_x, pred_mu.reshape(X.shape[0], X.shape[1], -1)), dim = 2)

            print(i,  MAPE(pred_mu, tmp_y), torch.mean((pred_mu - tmp_y)**2))

    def eval_prediction(self, X, y):
        if self.double_pre:
            y = y.double().to(device)
        else:
            y = y.float().to(device)
        pred_mu, pred_sig = self(X, prediction = True)
        pred_mu = pred_mu.reshape(y.shape)
        print(pred_mu[0], y[0])
        return MAPE(pred_mu, y), torch.mean(pred_sig), torch.mean((pred_mu - y)**2), torch.mean(y**2), torch.mean(pred_mu**2)

def learn_SGD_WFA(model_params, DATA, initial_bias = None):
    lr, epochs, batch_size, double_precision = model_params['fine_tune_lr'], model_params['fine_tune_epochs'], model_params[
        'batch_size'], model_params['double_precision']
    d, xd, r, mixture_n = model_params['d'], model_params['xd'], model_params['r'], model_params['mixture_n']
    l = model_params['l']
    verbose = model_params['verbose']
    generator_params = {'batch_size': batch_size,
                        'shuffle': True,
                        'num_workers': 0}
    dwfa = density_wfa(xd, d, r, mixture_n, device, use_batchnorm = True, init_std = 0.01, double_pre = False, initial_bias = initial_bias)
    merged_train = []
    merged_test = []
    Ls = [l, 2*l, 2*l+1]
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
    for k in range(len(Ls)):
        # print(data_label[k][0])
        train_x = DATA[data_label[k][0]]
        test_x = DATA[data_label[k][1]]
        merged_train.append(train_x)
        merged_test.append(test_x)
        train_loader = torch.utils.data.DataLoader(train_x, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_x, **generator_params)

        optimizer = optim.Adam(dwfa.parameters(), lr=lr, amsgrad=True)
        train_likeli, test_likeli = dwfa.fit(train_x, test_x, train_loader, test_loader, epochs,
                                                      optimizer, scheduler=None,
                                                      verbose=verbose)
    return dwfa

if __name__ == '__main__':
    load = False
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
    N = 1000  # number of training samples
    d = 1  # input encoder output dimension
    xd = 1  # input dimension
    r = 3  # rank/number of mixtures
    l = 5  # length in trianning (l, 2l, 2l+1)
    Ls = [l, 2 * l, 2 * l + 1]
    np.random.seed(1993)
    mixture_n = r

    model_params = {
        'd': d,
        'xd': xd,
        'r': r,
        'mixture_n': mixture_n,
        'lr': 0.1,
        'epochs': 1,
        'batch_size': 256,
        'fine_tune_epochs': 50,
        'regression_lr': 0.001,
        'regression_epochs': 1,
        'fine_tune_lr': 0.1,
        'double_precision': False,
        'verbose': True,
        'nn_transition': False,
        'GD_linear_transition': False,
        'use_softmax_norm': True,
        'l': l
    }
    train_x, test_x, hmmmodel = ggh.gen_gmmhmm_data(N=1, xd=xd, L=l, r=r)

    DATA = {}

    for k in range(len(Ls)):
        L = Ls[k]
        train_x = np.zeros([N, xd, L])
        test_x = np.zeros([N, xd, L])
        for i in range(N):
            x, z = hmmmodel.sample(L)
            train_x[i, :, :] = x.reshape(xd, -1)
            x, z = hmmmodel.sample(L)
            test_x[i, :, :] = x.reshape(xd, -1)

        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()
        DATA[data_label[k][0]] = train_x
        DATA[data_label[k][1]] = test_x

    bias = torch.mean(train_x.reshape(train_x.shape[0], -1), dim = 1)
    dwfa_finetune = learn_SGD_WFA(model_params, DATA, initial_bias=bias)

    N = 1000
    ls = [3, 10, 50,100, 200]
    for l in ls:
        train_x = np.zeros([N, xd, 2 * l])
        # print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2 * l)
            train_x[i, :, :] = x.reshape(xd, -1)

        test_x = np.zeros([N, xd, 2 * l+1])
        # print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2 * l+1)
            test_x[i, :, :] = x.reshape(xd, -1)

        train_ground_truth = ground_truth_hmm(train_x, hmmmodel)
        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()
        likelihood = dwfa_finetune.eval_likelihood(train_x)
        print("Length" + str(2 * l) + "result is:")
        print("Model output: " + str(np.mean(likelihood)) + "Ground truth: " + str(train_ground_truth))
        print(dwfa_finetune.eval_prediction(train_x[:, :, :-1], train_x[:, :, -1]))
        print(MAPE(torch.ones(train_x[:, :, -1].shape), train_x[:, :, -1]))