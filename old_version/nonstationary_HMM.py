import numpy as np
import torch
from torch import nn
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from Density_WFA_finetune import learn_density_WFA
from hmmlearn import hmm

class incremental_HMM(nn.Module):

    def __init__(self, r):
        super().__init__()
        self.transition = torch.rand([r, r])
        self.transition = torch.softmax(self.transition, dim = 1)
        self.sig = torch.ones(r).reshape([1, r])
        self.init_w = torch.rand([1, r])
        self.init_w = torch.softmax(self.init_w, dim = 1)
        self.mu_rates = (torch.arange(r)*0.1).reshape([1, r])


    def torch_mixture_gaussian(self, mu, sig, h):
        mix = D.Categorical(h)
        comp = D.Normal(mu, sig)
        gmm = mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm

    def sample_from_gmm(self, gmm, n):
        # print(gmm.sample([n]))
        return gmm.sample([n])

    def evalue_log_likelihood(self, X):
        h = self.init_w
        log_prob = 0.
        for i in range(X.shape[2]):
            mu = i*self.mu_rates
            gmm = self.torch_mixture_gaussian(mu, self.sig, h)
            log_prob += gmm.log_prob(X[:, :, i])
            h = h @ self.transition
        return torch.mean(log_prob)

    def sample(self, n, l):
        h = self.init_w
        samples = torch.zeros([n, 1, l])
        for i in range(l):
            mu = i * self.mu_rates
            gmm = self.torch_mixture_gaussian(mu, self.sig, h)
            current_sample = self.sample_from_gmm(gmm, n)
            samples[:, :, i] = current_sample
            h = h @ self.transition
        return samples

def fit_gmm(X, lens, r):
    remodel = hmm.GaussianHMM(n_components=r, covariance_type="full", n_iter=100)
    remodel.fit(X, lens)
    return remodel


if __name__ == '__main__':
    load = False
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]

    N = 10000  # number of training samples
    d = 3  # input encoder output dimension
    xd = 1  # input dimension
    r = 3  # rank/number of mixtures
    l = 3  # length in trianning (l, 2l, 2l+1)
    Ls = [l, 2 * l, 2 * l + 1]
    mixture_n = r

    model_params = {
        'd': 3,
        'xd': 1,
        'r': r,
        'mixture_n': mixture_n,
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 256,
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.001,
        'double_precision': False
    }


    nshmm = incremental_HMM(r)

    DATA = {}

    first = True
    hmm_train = 0
    for k in range(len(Ls)):
        L = Ls[k]
        train_x =nshmm.sample(N, L)
        test_x = nshmm.sample(N, L)
        if first:
            hmm_train = train_x.numpy().ravel().reshape(-1, 1)
            lens = np.ones(train_x.shape[0])*train_x.shape[2]
            first = False
        else:
            hmmm_train_new = train_x.numpy().ravel().reshape(-1, 1)
            lens_new = np.ones(train_x.shape[0]) * train_x.shape[2]
            hmm_train = np.concatenate([hmm_train, hmmm_train_new])
            lens = np.concatenate([lens, lens_new])


        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()
        DATA[data_label[k][0]] = train_x
        DATA[data_label[k][1]] = test_x

    print(hmm_train.shape)

    hmm_model  = fit_gmm(hmm_train, lens.astype(int), r)

    dwfa_finetune = learn_density_WFA(DATA, model_params, l)

    ls = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in ls:
        train_x = nshmm.sample(N, 2*l)
        train_ground_truth = nshmm.evalue_log_likelihood(train_x)
        train_x = torch.tensor(train_x).float()

        hmm_train = train_x.numpy().ravel().reshape(-1, 1)
        lens = np.ones(train_x.shape[0]) * train_x.shape[2]

        likelihood = dwfa_finetune.eval_likelihood(train_x)
        likelihood_hmm = hmm_model.score(hmm_train, lens.astype(int))
        likelihood = torch.tensor(likelihood)
        print("Length" + str(2 * l) + "result is:")
        print("wfa output: " + str(torch.mean(likelihood)) + "Ground truth: " + str(train_ground_truth) + 'HMM results: '+str(likelihood_hmm))


