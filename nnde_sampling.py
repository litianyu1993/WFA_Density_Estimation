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
from utils import phi, encoding, Fnorm, get_parameters_dist
from scipy.stats import norm

# from density_estimator import Hankel, evaluate_gaussian_hmm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# torch.autograd.set_detect_anomaly(True)


class hankel_density(nn.Module):

    def __init__(self, d, xd, r, nade_hid=None, mixture_number=10, L=5, init_std=1e-3, device=device, double_pre=True,
                 previous_hd=None, encoder_hid=None, nn_transition=False, GD_linear_transition=False,
                 train_encoder_termination=False, dummy=False):
        super().__init__()
        self.dummy = dummy
        if dummy:
            d = d + 1
        self.GD_linear_transition = GD_linear_transition
        if nade_hid is None:
            self.nade_hid = [128]
        else:
            self.nade_hid = nade_hid
        self.nn_transition = nn_transition
        if encoder_hid is None:
            self.encoder_hid = [128]
        else:
            self.encoder_hid = encoder_hid

        self.device = device
        self.core_list = nn.ParameterList()
        self.rank = r
        self.input_dim = d
        self.length = L
        self.mixture_number = mixture_number

        tmp_core = torch.normal(0, init_std, [1, r])
        self.init_w = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        tmp_core_back = torch.ones([1, r])
        self.init_w_back = nn.Parameter(tmp_core_back.clone().float().requires_grad_(False))
        for i in range(L):
            # if i == 0:
            #     tmp_core = nn.Parameter(torch.normal(0, init_std, [d, r]).clone().float().requires_grad_(True))
            # else:
            tmp_core = torch.normal(0, init_std, [r, d, r])
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

            self.mu_out = torch.nn.Linear(self.nade_hid[-1], mixture_number * xd, bias=True)
            self.sig_out = torch.nn.Linear(self.nade_hid[-1], mixture_number * xd, bias=True)
            self.alpha_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)

            self.encoder_1 = torch.nn.Linear(xd, self.encoder_hid[0], bias=True)
            self.encoder_2 = torch.nn.Linear(self.encoder_hid[0], d, bias=True)

        else:
            self.mu_out = previous_hd.mu_out
            self.mu_out.requires_grad = train_encoder_termination

            self.sig_out = previous_hd.mu_out
            self.sig_out.requires_grad = train_encoder_termination

            self.alpha_out = previous_hd.alpha_out
            self.alpha_out.weight.requires_grad = train_encoder_termination
            self.alpha_out.bias.requires_grad = train_encoder_termination

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
        if self.dummy:
            ori_X = X.clone()
            ori_X = ori_X[:, :-1, :]
        result = 0.
        norm = 0.
        for i in np.flip(np.arange(X.shape[2])):
            if i == X.shape[2] - 1:
                # images_repeated = images_vec.repeat(1, sequence_length)
                tmp = self.init_w_back.repeat(X.shape[0], 1)
                # print(i, tmp.shape)
            else:
                # print(i, tmp.shape, self.core_list[i - 1].shape, self.core_list[0].shape)
                tmp = torch.einsum("nd, nj, idj -> ni", encoding(self, X[:, :, i + 1]), tmp, self.core_list[i + 1])
            tmp_result = phi(self, ori_X[:, :, i], tmp)
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
        if self.dummy:
            ori_X = X.clone()
            ori_X = ori_X[:, :-1, :]

        result = 0.
        norm = 0.
        for i in range(self.length):
            if i == 0:
                tmp = self.init_w.repeat(X.shape[0], 1)
                # print(i, tmp.shape)
            else:
                # print(i, tmp.shape, self.core_list[i - 1].shape, self.core_list[0].shape)
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, self.core_list[i - 1])
            tmp_result = phi(self, ori_X[:, :, i], tmp)
            norm += Fnorm(tmp)
            result = result + tmp_result
        return result, norm

    def sampling(self, X, pos):
        if self.double_pre:
            X = X.double().to(device)
            dummy_vec = torch.zeros(X.shape[1]).double().to(self.device)
            dummy_vec[-1] = 1
        else:
            X = X.float().to(device)
            dummy_vec = torch.zeros(X.shape[1]).float().to(self.device)
            dummy_vec[-1] = 1
        for i in range(pos):
            if i == 0:
                tmp = self.init_w.repeat(X.shape[0], 1)
                # print(i, tmp.shape)
            else:
                # print(i, tmp.shape, self.core_list[i - 1].shape, self.core_list[0].shape)
                # print(i, self.core_list[i-1].shape)
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, self.core_list[i - 1])
        prev_tmp = tmp
        for i in np.flip(np.arange(X.shape[2])):
            if i == pos+1: break
            if i == X.shape[2] - 1:
                tmp = self.init_w_back.repeat(X.shape[0], 1)
            else:
                tmp = torch.einsum("nd, nj, idj -> ni", encoding(self, X[:, :, i + 1]), tmp, self.core_list[i + 1])

        curr_core = torch.einsum('ijk, j->ik', self.core_list[pos], dummy_vec)
        suff_tmp = torch.transpose((curr_core @ torch.transpose(tmp, 0, 1)), 0, 1)

        # print(prev_tmp.shape, suff_tmp.shape)
        tmp = torch.mul(prev_tmp, suff_tmp)
        # print(get_parameters_dist(self, X[:, :-1, :], prev_tmp@curr_core))
        # print(X[:, :, pos])
        return get_parameters_dist(self, X[:, :-1, :], tmp)
    #
    def eval_sampling(self, sampled, real):
        print(sampled.shape, real.shape)
        sampled = torch.ravel(sampled).float().to(device)

        real = torch.ravel(real[:, :-1]).float().to(device)

        return (torch.sqrt(torch.mean((sampled - real)**2))), torch.sqrt(torch.mean(real**2))

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
            if epoch > 10 and validation_likelihood[-1] > validation_likelihood[-2]: count  +=1
            if count >10 : break
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])
        # if not self.nn_transition and not self.GD_linear_transition:
        #     if train_x.shape[1] == 1:
        #         self.core_list[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]))
        #     else:
        #         self.core_list[0] = nn.Parameter(
        #             torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]).squeeze())
        return train_likehood, validation_likelihood

    def eval_likelihood(self, X):
        # print(X)
        log_likelihood, hidden_norm = self(X)
        return torch.mean(log_likelihood)

    def lossfunc(self, X):
        log_likelihood, hidden_norm = self(X)
        backward_likelihood, backward_norm = self.backward(X)
        log_likelihood = torch.mean(log_likelihood)
        backward_likelihood = torch.mean(backward_likelihood)
        hidden_norm = torch.mean(hidden_norm)
        # sum_trace = 0.
        # for i in range(1, self.length):
        #     for j in range(self.core_list[i].shape[1]):
        #         sum_trace += torch.sqrt(torch.trace(self.core_list[i][:, j, :]) ** 2)
        # print(self(X))
        return -log_likelihood + (backward_likelihood - log_likelihood) ** 2


def ground_truth_hmm(X, hmmmodel):
    log_likelihood = []
    for i in range(X.shape[0]):
        # p = hmmmodel.predict_proba(X[i, :, :].transpose())
        p = hmmmodel.score(X[i, :, :].transpose())
        log_likelihood.append(p)
    log_likelihood = np.asarray(log_likelihood)
    tmp = log_likelihood
    # print(tmp)
    return np.mean(tmp)


def insert_dummy_dim(X):
    bias = torch.zeros([X.shape[0], 1, X.shape[2]])
    X_bias = torch.cat((X, bias), 1)
    return X_bias


def random_remove_input_timestep(x):
    X = x.clone()
    for i in range(X.shape[0]):
        ran = np.random.randint(0, X.shape[2], size=1)[0]
        X[i, :, ran] *= 0
        X[i, -1, ran] = 1
    return X


def create_dummy_dim_data(X):
    X = insert_dummy_dim(X)
    dummy_X1 = random_remove_input_timestep(X)
    dummy_X2 = random_remove_input_timestep(X)
    dummy_X3 = random_remove_input_timestep(X)
    X = torch.cat((X, dummy_X1, dummy_X2, dummy_X3), 0)
    r = torch.randperm(X.shape[0])
    return X[r]


if __name__ == "__main__":
    np.random.seed(1993)
    N = 1000
    d = 1
    xd = 1
    r = 3
    L = 3
    mixture_n = 3

    batch_size = 256
    lr = 0.01
    epochs = 100

    train_x, test_x, hmmmodel = ggh.gen_gmmhmm_data(N=N, xd=xd, L=L)
    print(hmmmodel.means_, hmmmodel.covars_)
    test_ground_truth = ggh.ground_truth(hmmmodel, test_x)
    train_ground_truth = ggh.ground_truth(hmmmodel, train_x)
    print(train_ground_truth, test_ground_truth)

    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    train_x = create_dummy_dim_data(train_x)
    test_x = insert_dummy_dim(test_x)
    # print(train_x[0])
    # print(train_x[-1])
    # d = d+1
    # xd = xd+1


    hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, dummy=True)
    train_data = Dataset(data=[train_x])
    test_data = Dataset(data=[test_x])

    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 2}
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

    optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)
    likelihood, _ = hd(train_x)
    print(torch.mean(torch.log(likelihood)))
    train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer, scheduler=None,
                                       verbose=True)
    print(test_ground_truth, train_ground_truth, train_likeli[-1], test_likeli[-1])
    sampled = hd.sampling(test_x, pos = 2)
    print(hd.eval_sampling(sampled, test_x[:, :, 2]))
    plt.plot(-np.asarray(train_likeli), label='train')
    plt.plot(-np.asarray(test_likeli), label='test')
    plt.plot(test_ground_truth * np.ones(len(test_likeli)), label='test truth')
    plt.plot(train_ground_truth * np.ones(len(train_likeli)), label='train truth')
    plt.legend()
    plt.show()