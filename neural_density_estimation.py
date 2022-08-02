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


# torch.autograd.set_detect_anomaly(True)


class hankel_density(nn.Module):

    def __init__(self, d, xd, r, nade_hid=None, mixture_number=10, L=5, init_std=1e-3, device=device, double_pre=True,
                 previous_hd=None, encoder_hid=None, nn_transition = False, GD_linear_transition = False, train_encoder_termination = True, use_softmax_norm = False):
        super().__init__()
        self.use_softmax_norm = use_softmax_norm
        self.GD_linear_transition = GD_linear_transition
        if nade_hid is None:
            self.nade_hid = [128]
        else:
            self.nade_hid = nade_hid
        self.nn_transition = nn_transition
        if encoder_hid is None:
            self.encoder_hid = [d]
        else:
            self.encoder_hid = encoder_hid

        self.device = device
        self.core_list = nn.ParameterList()
        self.rank = r
        self.input_dim = d
        self.length = L
        self.mixture_number = mixture_number

        tmp_core_back = torch.normal(0, init_std, [1, r])
        self.init_w_back = nn.Parameter(tmp_core_back.clone().float().requires_grad_(True))
        for i in range(L):
            tmp_core = torch.normal(0., init_std, [r, d, r])
            tmp_core = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
            self.core_list.append(tmp_core)

        self.nade_layers = nn.ModuleList()
        nade_hid = [r] + self.nade_hid + [r]
        for i in range(len(nade_hid) - 1):
            first = nade_hid[i]
            next = nade_hid[i + 1]
            tmp_layer = torch.nn.Linear(first, next)
            self.nade_layers.append(tmp_layer)
        self.nade_hid = nade_hid
        # print(nade_hid)
        # if

        if previous_hd is None:
            # tmp_core = torch.normal(0, init_std, [self.nade_hid[-1], mixture_number, xd])
            # self.mu_out = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
            # self.mu_out_bias = nn.Parameter(torch.normal(0, init_std, [mixture_number, xd]).float().requires_grad_(True))
            #
            # tmp_core = torch.normal(0, init_std, [self.nade_hid[-1], mixture_number, xd])
            # self.sig_out = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
            # self.sig_out_bias = nn.Parameter(
            #     torch.normal(0, init_std, [mixture_number, xd]).float().requires_grad_(True))
            self.batchnrom = nn.BatchNorm1d(r)
            self.mu_out = torch.nn.Linear(self.nade_hid[-1], mixture_number*xd, bias=True)
            self.sig_out = torch.nn.Linear(self.nade_hid[-1], mixture_number*xd, bias=True)
            self.alpha_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)


            self.encoder_1 = torch.nn.Linear(xd, self.encoder_hid[0], bias=True)
            self.encoder_2 = torch.nn.Linear(self.encoder_hid[0], d, bias=True)
            tmp_core = torch.normal(0, init_std, [1, r])
            self.init_w = nn.Parameter(tmp_core.clone().float().requires_grad_(True))

        else:
            for i in range(len(previous_hd.core_list)):
                self.core_list[i] = previous_hd.core_list[i].requires_grad_(True)
            self.batchnrom = previous_hd.batchnrom
            self.batchnrom.requires_grad = train_encoder_termination
            self.init_w = previous_hd.init_w.requires_grad_(True)
            self.init_w.requires_grad = train_encoder_termination
            self.mu_out = previous_hd.mu_out
            self.mu_out.requires_grad = train_encoder_termination

            self.sig_out = previous_hd.mu_out
            self.sig_out.requires_grad = train_encoder_termination

            self.alpha_out = previous_hd.alpha_out
            self.alpha_out.weight.requires_grad = train_encoder_termination
            # self.alpha_out.bias.requires_grad = train_encoder_termination

            self.encoder_1 = previous_hd.encoder_1
            self.encoder_2 = previous_hd.encoder_2
            self.encoder_1.weight.requires_grad = train_encoder_termination
            self.encoder_1.bias.requires_grad = train_encoder_termination
            self.encoder_2.weight.requires_grad = train_encoder_termination
            self.encoder_2.bias.requires_grad = train_encoder_termination

        self.double_pre = double_pre
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)



    def backward(self, X):
        # print(X.shape[2], self.length)
        assert X.shape[2] == self.length, "trajectory length does not fit the network structure"
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)

        result = 0.
        norm = 0.
        for i in np.flip(np.arange(X.shape[2])):
            if i == X.shape[2] - 1:
                # images_repeated = images_vec.repeat(1, sequence_length)
                tmp = self.init_w_back.repeat(X.shape[0], 1)
                # print(i, tmp.shape)
            else:
                # print(i, tmp.shape, self.core_list[i - 1].shape, self.core_list[0].shape)
                tmp = torch.einsum("nd, nj, idj -> ni", encoding(self, X[:, :, i + 1]), tmp, self.core_list[i+1])

            tmp_result = phi(self, X[:, :, i], tmp)

            norm += Fnorm(tmp)
            result = result + tmp_result
        return result, norm

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
                # if self.use_softmax_norm:
                #     tmp = torch.softmax(tmp, dim = 1)
            else:
                # print(i, tmp.shape, self.core_list[i - 1].shape, self.core_list[0].shape)
                tmp_A = self.core_list[i - 1]
                # if self.use_softmax_norm:
                #     tmp_A = torch.softmax(self.core_list[i - 1], dim  = 2)
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, tmp_A)
                if self.use_softmax_norm:
                    tmp = self.batchnrom(tmp)
            tmp_result = phi(self, X[:, :, i], torch.softmax(tmp, dim = 1))
            norm += Fnorm(tmp)
            result = result + tmp_result
        return result, norm


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
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])
        # self.init_w = nn.Parameter(torch.softmax(self.init_w, dim = 1))
        # for i in range(len(self.core_list)):
        #     self.core_list[i] = nn.Parameter(torch.softmax(self.core_list[i], dim = 2))
        # self.use_softmax_forward = False

        if not self.nn_transition and not self.GD_linear_transition and not self.use_softmax_norm:
            # print(train_x.shape, self.init_w.shape, self.core_list[0].shape, self.nn_transition, self.GD_linear_transition)
            if self.input_dim == 1:
                self.core_list[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]).reshape(1, self.rank))
            else:
                self.core_list[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]).squeeze())

        return train_likehood, validation_likelihood

    def eval_likelihood(self, X):
        # print(X)
        log_likelihood, hidden_norm = self(X)
        return torch.mean(log_likelihood)

    def lossfunc(self, X):
        log_likelihood, hidden_norm = self(X)
        # backward_likelihood, backward_norm = self.backward(X)
        log_likelihood = torch.mean(log_likelihood)
        # backward_likelihood = torch.mean(backward_likelihood)
        hidden_norm = torch.mean(hidden_norm)
        # sum_trace = 0.
        # for i in range(1, self.length):
        #     for j in range(self.core_list[i].shape[1]):
        #         sum_trace += torch.sqrt(torch.trace(self.core_list[i][:, j, :]) ** 2)
        # print(self(X))
        return -log_likelihood


def ground_truth_hmm(X, hmmmodel):
    log_likelihood = []
    for i in range(X.shape[0]):
        # p = hmmmodel.predict_proba(X[i, :, :].transpose())
        p = hmmmodel.score(X[i, :, :].transpose())
        log_likelihood.append(p)
    log_likelihood = np.asarray(log_likelihood)
    tmp = log_likelihood
    return np.mean(tmp)


def insert_dummy_dim(X):
    bias = torch.zeros([X.shape[0], 1, X.shape[2]])
    X_bias = torch.cat((X, bias), 1)
    return X_bias

def random_remove_input_timestep(x):
    X = x.clone()
    for i in range(X.shape[0]):
        ran = np.random.randint(0, X.shape[2], size = 1)[0]
        X[i, :, ran] *= 0
        X[i, -1, ran] = 1
    return X

def create_dummy_dim_data(X):
    X = insert_dummy_dim(X)
    dummy_X = random_remove_input_timestep(X)
    X = torch.cat((X, dummy_X), 0)
    return X

if __name__ == "__main__":

    N = 1000
    d = 5
    xd = 1
    r = 3
    L = 10
    mixture_n = 3

    batch_size = 256
    lr = 0.01
    epochs = 50

    train_x, test_x, hmmmodel = ggh.gen_gmmhmm_data(N = N, xd = xd, L = L)
    test_ground_truth = ggh.ground_truth(hmmmodel, test_x)
    train_ground_truth =  ggh.ground_truth(hmmmodel, train_x)
    print(train_ground_truth, test_ground_truth)

    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    # train_x = insert_dummy_dim(train_x)
    # test_x = insert_dummy_dim(test_x)
    # print(train_x[0])
    # print(train_x[-1])
    # d = d+1
    # xd = xd+1

    print(train_x.shape)

    hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L)
    train_data = Dataset(data=[train_x])
    test_data = Dataset(data=[test_x])

    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 0}
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

    optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)
    likelihood, _ = hd(train_x)
    print(torch.mean(torch.log(likelihood)))
    train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer, scheduler=None,
                                       verbose=True)
    print(test_ground_truth, train_ground_truth, train_likeli[-1], test_likeli[-1])
    plt.plot(-np.asarray(train_likeli), label='train')
    plt.plot(-np.asarray(test_likeli), label='test')
    plt.plot(test_ground_truth * np.ones(len(test_likeli)), label='test truth')
    plt.plot(train_ground_truth * np.ones(len(train_likeli)), label='train truth')
    plt.legend()
    plt.show()