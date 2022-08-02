import numpy as np
import torch
from torch import nn
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from Density_WFA_finetune import learn_density_WFA
from hmmlearn import hmm
import copy
class incremental_HMM(nn.Module):

    def __init__(self, r, seed = 1993):
        np.random.seed(seed)
        super().__init__()
        # print('current rank is ', r)
        self.transition = torch.rand([r, r])
        self.transition = torch.softmax(self.transition, dim = 1)
        self.sig = torch.rand(r).reshape([1, r])
        self.init_w = torch.rand([1, r])
        self.init_w = torch.softmax(self.init_w, dim = 1)
        self.mu_rates = torch.rand(r).reshape([1, r])
        np.random.seed(seed)
        self.seed = seed
        torch.manual_seed(self.seed)


    def torch_mixture_gaussian(self, mu, sig, h):
        mix = D.Categorical(h)
        comp = D.Normal(mu, sig)
        gmm = mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm

    def sample_from_gmm(self, gmm, n):
        # print(gmm.sample([n]))

        return gmm.sample([n])


    def score(self, X, stream = False):
        import copy
        h = self.init_w
        joint = 0.
        if stream:
            all_probs = []
            all_conditionals = []
        mu = copy.deepcopy(self.mu_rates)
        for i in range(X.shape[0]):
            # if i % 300 == 0:
            #     mu+=10
            gmm = self.torch_mixture_gaussian(mu, self.sig, h)
            tmp = gmm.log_prob(torch.tensor(X[i, :]))
            joint += tmp
            h = h @ self.transition
            if stream:
                all_probs.append(copy.deepcopy(joint.numpy()))
                all_conditionals.append(tmp.numpy())
            # print(i, h)
            # print(i, 'scoring', mu.reshape(1, -1)@h.reshape(-1, 1))
        if stream:
            return all_probs, all_conditionals
        else:
            return joint.numpy()



    def sample(self, l, seed = 1993):
        np.random.seed(seed)
        torch.manual_seed(seed)
        h = self.init_w
        samples = torch.zeros([1, l])
        mu = copy.deepcopy(self.mu_rates)
        for i in range(l):
            # if i % 300 ==0:
            #     mu += 10
            gmm = self.torch_mixture_gaussian(mu, self.sig, h)
            current_sample = self.sample_from_gmm(gmm, 1)
            samples[:, i] = current_sample
            h = h @ self.transition
            # print(i,'sampling', mu.reshape(1, -1)@h.reshape(-1, 1))
        return samples

def fit_gmm(X, lens, r):
    remodel = hmm.GaussianHMM(n_components=r, covariance_type="full", n_iter=100)
    remodel.fit(X, lens)
    return remodel


if __name__ == '__main__':
    load = False
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]

    N = 1  # number of training samples
    d = 3  # input encoder output dimension
    xd = 1  # input dimension
    r = 3  # rank/number of mixtures
    l = 1000  # length in trianning (l, 2l, 2l+1)
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


