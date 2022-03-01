import numpy as np
import torch
from torch import nn
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
from scipy.stats import norm
# from density_estimator import Hankel, evaluate_gaussian_hmm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# torch.autograd.set_detect_anomaly(True)

        

class hankel_density(nn.Module):

    def __init__(self, d, xd, r, nade_hid = None, mixture_number = 10, L = 5, init_std = 1e-3, device = device, double_pre = True, previous_hd = None, encoder_hid = None):
        super().__init__()
        if nade_hid is None:
            self.nade_hid = [128]
        else:
            self.nade_hid = nade_hid

        if encoder_hid is None:
            self.encoder_hid = [128]
        else:
            self.encoder_hid = encoder_hid

        self.device = device
        self.core_list = nn.ParameterList()
        self.rank = r
        self.input_dim = d
        self.length  = L
        self.mixture_number = mixture_number

        tmp_core = torch.normal(0, init_std, [1, r])
        self.init_w = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        for i in range(L):
            # if i == 0:
            #     tmp_core = nn.Parameter(torch.normal(0, init_std, [d, r]).clone().float().requires_grad_(True))
            # else:
            tmp_core = torch.normal(0, init_std, [r, d, r])
            tmp_core = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
            self.core_list.append(tmp_core)

        self.nade_layers = nn.ModuleList()
        nade_hid = [r] + self.nade_hid
        for i in range(len(nade_hid) - 1):
            first = nade_hid[i]
            next = nade_hid[i + 1]
            tmp_layer = torch.nn.Linear(first, next)
            self.nade_layers.append(tmp_layer)

        # if
        if previous_hd is None:
            self.mu_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)
            self.sig_out= torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)
            self.alpha_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)

            self.encoder_1 = torch.nn.Linear(xd, self.encoder_hid[0], bias=True)
            self.encoder_2 = torch.nn.Linear(self.encoder_hid[0], d, bias=True)

        else:
            self.mu_out = previous_hd.mu_out
            self.mu_out.weight.requires_grad = False
            self.mu_out.bias.requires_grad = False


            self.sig_out = previous_hd.sig_out
            self.sig_out.weight.requires_grad = False
            self.sig_out.bias.requires_grad = False

            self.alpha_out = previous_hd.alpha_out
            self.alpha_out.weight.requires_grad = False
            self.alpha_out.bias.requires_grad = False

            self.encoder_1 = previous_hd.encoder_1
            self.encoder_2 = previous_hd.encoder_2
            self.encoder_1.weight.requires_grad = False
            self.encoder_1.bias.requires_grad = False
            self.encoder_2.weight.requires_grad = False
            self.encoder_2.bias.requires_grad = False

        # if mu_out is None:
        #     self.mu_out1 = torch.nn.Linear(self.nade_hid[-1] , mixture_number, bias=True)
        #     self.mu_out2 = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=True)
        # else:
        #     self.mu_out1= mu_out
        #     self.mu_out.weight.requires_grad = False
        #     self.mu_out.bias.requires_grad = False
        #
        # if sig_out is None:
        #     self.sig_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias = False)
        # else:
        #     self.sig_out = sig_out
        #     self.sig_out.weight.requires_grad = False
        #
        # if alpha_out is None:
        #     self.alpha_out = torch.nn.Linear(self.nade_hid[-1], mixture_number, bias=False)
        # else:
        #     self.alpha_out = alpha_out
        #     self.alpha_out.weight.requires_grad = False

        self.double_pre = double_pre
        if double_pre:
            self.double().cuda()
        else:
            self.cuda()

    def torch_mixture_gaussian(self, X, mu, sig, alpha):
        mix = D.Categorical(alpha)
        comp = D.Normal(mu, sig)
        gmm = mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm.log_prob(X)

    def phi(self, X, h):
        # print(h.shape)
        h = torch.tanh(h)
        for i in range(len(self.nade_layers)):
            h = self.nade_layers[i](h)
            h = torch.relu(h)
        mu = self.mu_out(h)
        sig = torch.exp(self.sig_out(h))
        # print(torch.mean(sig))
        alpha = torch.softmax(self.alpha_out(h), dim = 1)
        return self.torch_mixture_gaussian(X, mu, sig, alpha)

    # def gaussian(self, X, mu, sig):
    #
    #     # print("X",X.shape, mu.shape, sig.shape)
    #     result = (X - mu) ** 2
    #     # print(result[0])
    #     tmp_sig = 2*sig**2
    #     result = -torch.div(result, tmp_sig)
    #     pi = torch.acos(torch.zeros(1)).item() * 2
    #     normalizer = 1. / (((2 * pi) ** 0.5) * sig)
    #     return torch.mul(normalizer,torch.exp(result))
    #
    # def mixture_gaussian(self, X, mu, sig, alpha):
    #     result = 0.
    #     for i in range(mu.shape[1]):
    #         tmp_result = self.gaussian(X, mu[:, i], sig[:, i])
    #         # print(mu[:, i].shape, sig[:, i].shape)
    #         # print(sig[:, i])
    #         # gassuain_models = Normal(mu[:, i], sig[:, i])
    #         # tmp_result = 10**(gassuain_models.log_prob(X))
    #         # print(X.shape, tmp_result.shape)
    #         result =result + torch.mul(alpha[:, i], tmp_result)
    #         # print(tmp_result[0])
    #
    #     return result

    def encoding(self, X):
        X = self.encoder_1(X)
        X = torch.relu(X)
        X = self.encoder_2(X)
        return torch.tanh(X)

    def forward(self, X):
        assert X.shape[2] == self.length, "trajectory length does not fit the network structure"
        if self.double_pre:
            X = X.double().cuda()
        else:
            X = X.cuda()

        result = 0.

        for i in range(self.length):
            if i == 0:
                tmp = self.init_w
            else:
                tmp = torch.einsum("nd, ni, idj -> nj", self.encoding(X[:, :, i-1]), tmp, self.core_list[i-1])
            # print(result, self.phi(X, tmp).shape)
            tmp_result = self.phi(X[:, :, i].squeeze(), tmp)
            # print(i, tmp_result)
            result = result + tmp_result
        return torch.exp(result)



    def negative_log_likelihood(self, X):
        log_likelihood = torch.mean(torch.log(self(X)))
        sum_trace = 0.
        for i in range(1, self.length):
            for j in range(self.core_list[i].shape[1]):
                sum_trace += torch.sqrt(torch.trace(self.core_list[i][:, j, :]) ** 2)
        # print(self(X))
        return -log_likelihood+sum_trace

    def fit(self,train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler = None, verbose = True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
            train_likehood.append(train(self, self.device, train_loader, optimizer, X = train_x).detach().to('cpu'))
            validation_likelihood.append(validate(self, self.device, validation_loader, X =test_x).detach().to('cpu'))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(train_likehood[-1],
                                                                                                     validation_likelihood[-1]))
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])

            # if epoch >3:
            #     if train_likehood[-1] - train_likehood[-2] > train_likehood[-2] - train_likehood[-3] and train_likehood[-1] - train_likehood[-2]>0:
            #         count += 1
            #
            # if count > 2: break

        self.core_list[0] = nn.Parameter(torch.einsum('ij, jkl -> kl', self.init_w, self.core_list[0]).squeeze())
        return train_likehood, validation_likelihood

    def eval_likelihood(self, X):
        # print(X)
        likelihood = (self(X)) #sum_to_one
        return torch.mean(torch.log(likelihood))



def ground_truth_hmm(X, hmmmodel):
    likelihood = []
    for i in range(X.shape[0]):
        # p = hmmmodel.predict_proba(X[i, :, :].transpose())
        p = np.exp(hmmmodel.score(X[i, :, :].transpose()))
        likelihood.append(p)
    likelihood = np.asarray(likelihood)
    # tmp = likelihood/np.sum(likelihood)
    tmp = np.log(likelihood)
    # print(tmp)
    return np.mean(tmp)

def insert_bias(X):
    bias = torch.ones([X.shape[0], 1, X.shape[2]])
    X_bias = torch.cat((X, bias), 1)
    return X_bias

if __name__ == "__main__":
    N = 10000
    d = 3
    xd = 1
    r = 3
    L = 7
    mixture_n = 3

    batch_size = 256
    lr = 0.001
    epochs = 500

    hmmmodel = hmm.GaussianHMM(n_components=3, covariance_type="full")
    hmmmodel.startprob_ = np.array([0.6, 0.3, 0.1])
    hmmmodel.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])
    hmmmodel.means_ = np.array([[0.0], [3.0], [5.0]])
    hmmmodel.covars_ = np.tile(np.identity(1), (3, 1, 1))
    train_x = np.zeros([N, xd, L])
    test_x = np.zeros([N, xd, L])



    print(train_x.shape)
    for i in range(N):
        x, z = hmmmodel.sample(L)

        train_x[i, :, :] = x.reshape(xd, L)
        x, z = hmmmodel.sample(L)
        test_x[i, :, :] = x.reshape(xd, L)
    print(train_x.shape)

    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    hmm_train_x = np.concatenate([train_x[i, :, :].transpose() for i in range(N)])
    hmm_length = train_x.shape[2] * N
    remodel.fit(hmm_train_x, hmm_length)
    print("hmm", remodel.score(hmm_train_x, hmm_length) / N)

    # test_hmm_x = np.zeros([N, xd, 2*L])
    # for i in range(N):
    #     x, z = hmmmodel.sample(2*L)
    #     test_hmm_x[i, :, :] = x.reshape(xd, 2*L)
    # hmm_test_x = np.concatenate([test_hmm_x[i, :, :].transpose() for i in range(N)])
    # hmm_length = test_hmm_x.shape[2] * N
    # print("hmm", remodel.score(hmm_test_x, hmm_length) / N, ground_truth_hmm(test_hmm_x, hmmmodel))



    test_ground_truth = ground_truth_hmm(test_x, hmmmodel)
    train_ground_truth = ground_truth_hmm(train_x, hmmmodel)

    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    print(train_x.shape)

    hd = hankel_density(d, xd, r, mixture_number=mixture_n, L = L)
    train_data = Dataset(data=[train_x])
    test_data = Dataset(data=[test_x])

    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 2}
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

    optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)
    likelihood =hd(train_x)
    print(torch.mean(torch.log(likelihood)))
    train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer, scheduler=None, verbose=True)

    plt.plot(train_likeli, label = 'train')
    plt.plot(test_likeli, label = 'test')
    plt.plot(test_ground_truth*np.ones(len(test_likeli)), label = 'test truth')
    plt.plot(train_ground_truth * np.ones(len(train_likeli)), label='train truth')
    plt.legend()
    plt.show()





