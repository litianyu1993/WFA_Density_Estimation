import pandas
import torch
from torch import nn
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
import numpy as np
from hmmlearn import hmm
import pickle
import argparse
import os
import copy
import math
sp = nn.Softplus()
from sklearn.mixture import GaussianMixture
def get_lr(gamma, optimizer):
    return [group['lr'] * gamma
            for group in optimizer.param_groups]

def gaussian(x, mu, sig):
    n = 1
    d = x.shape[0]
    diag_sig = torch.diag(sig)
    diff = (x - mu)
    tmp1 = -d*n/2*np.log(2*torch.pi)
    tmp2 = -n/2 * torch.log(torch.det(diag_sig))
    tmp3 = -0.5*diff.reshape(1, -1)@torch.inverse(diag_sig)@ diff.reshape(-1, 1)
    log_likeli = tmp1+tmp2+tmp3
    return log_likeli
def mix_of_gaussian(x, alpha, mus, sigs):
    '''
    alpha: shape k
    mus: shape k by d
    sigs: shape k by d
    '''
    if mus.ndim ==1:
        mus= mus.reshape(-1, 1)
    if sigs.ndim ==1:
        sigs= sigs.reshape(-1, 1)
    alpha = alpha.reshape(-1,)
    log_likeli = 0.
    for i in range(len(mus)):
        mu = mus[i]
        sig = sigs[i]
        log_likeli += alpha[i] * gaussian(x, mu, sig)
    return log_likeli


def get_artificial_distribution(density_name, N = 2000, sample_n = 1000, seed = 1993):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if density_name == 'normal':
        mix = D.Categorical(torch.ones(1, ))
        comp = D.Normal(torch.ones(1, )*10, torch.ones(1, )*0.1)
    elif density_name == 'skewed':
        mix = D.Categorical(torch.tensor([1/5, 1/5, 3/5]))
        comp = D.Normal(torch.tensor([0, 0.5, 13/12]), torch.tensor([1, 4/9, 25/49]))
    elif density_name == 'strong_skewed':
        mix = []
        mu = []
        sig = []
        for l in range(8):
            mix.append(1/8)
            mu.append(3*((2/3)**l - 1))
            sig.append( (2/3)**(2*l))
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig)**0.5)
    elif density_name == 'bimodal':
        mix = D.Categorical(torch.tensor([0.5, 0.5]))
        comp = D.Normal(torch.tensor([-1., 1.]), torch.tensor([2/3, 2/3]))
    elif density_name == 'outlier':
        mix = D.Categorical(torch.tensor([0.1, 0.9]))
        comp = D.Normal(torch.zeros(2, ), torch.tensor([1, 0.1]))
    elif density_name == 'trimodal':
        mix = D.Categorical(torch.tensor([9/20, 9/20, 1/10]))
        mu = [-6/5, 6/5, 0]
        sig = [3/5, 3/5, 1/4]
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'smooth_comb':
        mix = []
        mu = []
        sig = []
        for l in range(6):
            mix.append(2**(5-l))
            mu.append((65 - 96*(0.5)**l)/21)
            sig.append( (32/63)/(2**l))
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'claw':
        mix = [0.5]
        mu = [0.]
        sig = [1.]
        for l in range(5):
            mix.append(1/10)
            mu.append(l/2 - 1)
            sig.append(0.1)
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'double_claw':
        mix = [0.49, 0.49]
        mu = [-1, 1]
        sig = [2/3, 2/3]
        for l in range(7):
            mix.append(1/350)
            mu.append((l-3)/2)
            sig.append(0.01)
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'sym_claw':
        mix = [0.5]
        mu = [0]
        sig = [1]
        for l in [-2, -1, 0, 1, 2]:
            mix.append(2**(1-l)/31)
            mu.append(l+0.5)
            sig.append(2**(-l)/10)
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'asym_double_claw':
        mix = [0.46, 0.46]
        mu = [-1, 1]
        sig = [2/3, 2/3]
        for l in [1, 2, 3]:
            mix.append(1/300)
            mu.append(-l/2)
            sig.append(0.01)
        for l in [1, 2, 3]:
            mix.append(7/300)
            mu.append(l/2)
            sig.append(0.07)
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'dis_comb':
        mix = [2/7, 2/7, 2/7, 1/21, 1/21, 1/21]
        mu = []
        sig = [2/7, 2/7, 2/7, 1/21, 1/21, 1/21]
        for l in [0, 1, 2]:
            mu.append( (12*l - 15)/7)
        for l in [8, 9, 10]:
            mu.append( (2*l)/7)
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'kurtotic':
        mix = [2/3, 1/3]
        mu = [0., 0.]
        sig = [1, 0.1]
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'sep_bimodal':
        mix = [0.5, 0.5]
        mu = [-1.5, 1.5]
        sig = [0.5, 0.5]
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))
    elif density_name == 'skew_bimodal':
        mix = [0.75, 0.25]
        mu = [0, 1.5]
        sig = [1, 1/3]
        mix = D.Categorical(torch.tensor(mix))
        comp = D.Normal(torch.tensor(mu), torch.tensor(sig))

    gmm = mixture_same_family.MixtureSameFamily(mix, comp)
    X = gmm.sample_n(N*sample_n)
    density = gmm.log_prob(X)
    # X,_ = gm.sample(N * sample_n)
    # gm.fit_predict(X)
    # print(X.shape)
    # density = gm.score_samples(X.reshape(-1, 1))
    # print(density.shape)
    # from matplotlib import pyplot as plt
    # plt.scatter(X, torch.exp(density))
    # plt.show()

    return X.numpy().reshape(N, sample_n), density.numpy().reshape(N, sample_n)


def get_copy_paste(n =10000, d = 2, lag = 5, seed = 1993):
    from scipy.stats import norm, multivariate_normal
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=[lag+1, d])
    new_X = []
    y = []
    std = np.eye(2)
    for i in range(lag+1):
        new_X.append(X[i])
        # y.append(multivariate_normal.logpdf(new_X[-1], mean = [0, 0], cov = std))
        y.append(0)
    for i in range(lag, n):
        loc = new_X[i][0] - new_X[i-lag][0]
        if loc >=0:
            # x = np.random.normal(-1, 1, size = d)
            x = multivariate_normal.rvs(mean = [0, 0], cov = std, size=1)
            y.append(multivariate_normal.logpdf(x, mean = [0, 0], cov = std))
            # y.append(1)
        else:
            x = multivariate_normal.rvs(mean = [0, -1], cov = std, size=1)
            y.append(multivariate_normal.logpdf(x, mean = [0, -1], cov = std))
            # y.append(0)
        new_X.append(x)

    new_X = np.asarray(new_X)
    y = np.asarray(y)
    print(new_X[100])
    print(new_X.shape, y.shape)
    return new_X, y

    # np.random.seed(seed)
    # X = np.random.normal(0, 1, size = [lag, d])
    # y = []
    # y_neg = 0
    # y_pos = 0
    # for i in range(n):
    #     if X[i][0] - X[i-lag][0]>= 0:
    #         y.append(1)
    #         y_pos += 1
    #     else:
    #         y.append(0)
    #         y_neg += 1
    # # X = X[lag:]
    # y = np.asarray(y)
    # print(y_pos, y_neg, y_pos/n)
    # print(naive_prediction(y))
    # return X, y

def naive_prediction(y):
    count = 0
    for i in range(1, len(y)):
        if y[i] == y[i-1]: count += 1
    print(count/len(y))

class copy_paste_density(nn.Module):
    def __init__(self, lag = 5, seed = 1993):
        np.random.seed(seed)
        super().__init__()
        self.sig = 1
        self.lag = lag
        self.mu = 0
        torch.manual_seed(self.seed)


class incremental_HMM(nn.Module):

    def __init__(self, r, seed = 1993):
        np.random.seed(seed)
        super().__init__()
        # print('current rank is ', r)
        self.transition = torch.rand([r, r])
        self.transition = torch.softmax(self.transition, dim = 1)

        self.transition2 = torch.normal(2, 1, [r, r])
        self.transition2 = torch.softmax(self.transition, dim=1)
        # print(self.transition)
        self.sig = torch.ones(r).reshape([1, r])
        self.init_w = torch.rand([1, r])
        self.init_w = torch.softmax(self.init_w, dim = 1)
        self.mu_rates = torch.ones(r).reshape([1, r])
        np.random.seed(seed)
        self.seed = seed
        torch.manual_seed(self.seed)


    def torch_mixture_gaussian(self, mu, sig, h):
        mix = D.Categorical(h)
        comp = D.Normal(mu, sig)
        gmm = mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm

    def sample_from_gmm(self, gmm, n):

        return gmm.sample([n])



    def score(self, X, stream = False):
        import copy
        h = self.init_w
        if stream:
            all_conditionals = []
        mu = copy.deepcopy(self.mu_rates)
        sig = copy.deepcopy(self.sig)
        density = np.zeros(X.shape)
        for i in range(X.shape[1]):
            if i % 5 == 0:
                mu += 1
                # sig += 1
            # mu =  mu +0.1 #* i%5 *10
            gmm = self.torch_mixture_gaussian(mu, sig, h)
            for j in range(X.shape[2]):
                tmp = gmm.log_prob(torch.tensor(X[:, i, j]))
                if stream:
                    density[0, i, j] = tmp
            # h = h @ self.transition
            # if i % 1 == 0 and i % 2 != 0:
            #     h = h @ self.transition
            # if i %2 ==0:
            #     h = h @ self.transition2

            # print(i, h)
            # print(i, 'scoring', mu.reshape(1, -1)@h.reshape(-1, 1))
        if stream:
            return density

    def sample(self, l, n= 10, seed = 1993):
        np.random.seed(seed)
        torch.manual_seed(seed)
        h = self.init_w

        samples = torch.zeros([1, l, n])
        mu = copy.deepcopy(self.mu_rates)
        sig = copy.deepcopy(self.sig)
        for i in range(l):
            if i % 5 == 0:
                mu += 1
                # sig += 1
            # mu = mu +0.1# * i%5 *10

            gmm = self.torch_mixture_gaussian(mu, sig, h)
            current_sample = self.sample_from_gmm(gmm, n).reshape(-1)
            samples[:, i, :] = current_sample
            # h = h @ self.transition
            # if i % 1 == 0 and i % 2 != 0:
            #     h = h @ self.transition
            # if i % 2 == 0:
            #     h = h @ self.transition2
            # print(i,'sampling', mu.reshape(1, -1)@h.reshape(-1, 1))
        return samples
def get_hmm(r = 3, N = 1000, n = 10):
    nshmm = incremental_HMM(r=r)
    X = nshmm.sample(l=N, n = n)
    # X = np.asarray(X).swapaxes(0, 1)
    ground_truth_conditionals = nshmm.score(X, stream = True)
    ground_truth_conditionals = np.asarray(ground_truth_conditionals)
    return X, ground_truth_conditionals

def sliding_window(X, window_size = 5):
    final_data = []
    for j in range(len(X) - 1 - window_size):
        tmp = X[j:j+window_size]
        final_data.append(tmp)
    return np.asarray(final_data).swapaxes(1, 2)
def get_sea_data():
    X = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'sea', 'SEA_training_data.csv'),
                      delimiter=',')
    y = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'sea', 'SEA_training_class.csv'), delimiter=',')
    return X, y

# def get_spam_data():
#     file = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'spam', 'spam.libsvm')
#     data = datasets.load_svmlight_file(file)
#     print(data[0].shape, data[1].shape)
#     return data[0], data[1]

def get_electronic_data():
    data = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'Elec2', 'elec2_data.dat1'),
                         skip_header=1,
                         skip_footer=1,
                         names=True,
                         dtype=float,
                         delimiter=' ')
    new_data = []
    for i in range(len(data)):
        x = []
        for j in range(len(data[i])):
            x.append(data[i][j])
        new_data.append(x)
    new_data = np.asarray(new_data)
    X = new_data

    data = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'Elec2', 'elec2_label.dat2'),
                         skip_header=1,
                         skip_footer=1,
                         names=True,
                         dtype=float,
                         delimiter=' ')
    new_data = []
    for i in range(len(data)):
        x = []
        for j in range(len(data[i])):
            x.append(data[i][j])
        new_data.append(x)
    new_data = np.asarray(new_data)
    Y = new_data
    return X, Y

def get_movingSquares():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'movingSquares', 'movingSquares.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial','movingSquares', 'movingSquares.labels'),
        delimiter=' ')
    return X, y
def get_interRBF():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'rbf', 'interchangingRBF.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'rbf', 'interchangingRBF.labels'),
        delimiter=' ')
    return X, y
def get_movingRBF():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'rbf', 'movingRBF.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'rbf', 'movingRBF.labels'),
        delimiter=' ')
    return X, y
def get_outdoor():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'outdoor', 'Outdoor-train.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'outdoor', 'Outdoor-train.labels'),
        delimiter=' ')
    return X, y

def get_hyperplane():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'hyperplane', 'rotatingHyperplane.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'hyperplane', 'rotatingHyperplane.labels'),
        delimiter=' ')
    return X, y

def get_letter():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'letter', 'letter-recognition.data'),
        delimiter=',')
    file_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'letter', 'letter-recognition.data')
    file1 = open(file_path, 'r')
    Lines = file1.readlines()
    y = []
    for line in Lines:
        y.append(line.strip()[0])
    X = X[:, 1:]

    for i in range(len(y)):
        y[i] = ord(y[i].lower()) - 97
    y = np.asarray(y).reshape(-1, 1)
    return X, y

def get_poker(two_classes = True):
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'poker', 'poker.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'poker', 'poker.labels'),
        delimiter=' ')
    # new_x = []
    # for i in range(X.shape[1]):
    #     n_values = int(np.max(X[:, i]) + 1)
    #     # print(X[:, i].astype(int))
    #     new_x.append(np.eye(n_values)[X[:, i].astype(int)])
    # X = np.concatenate(new_x, axis=1)
    if two_classes:
        num_per_class = {}
        for i in range(len(y)):
            if y[i] not in num_per_class:
                num_per_class[y[i]] = 0
            else:
                num_per_class[y[i]] += 1
        import operator
        index1 = max(num_per_class.items(), key=operator.itemgetter(1))[0]
        del num_per_class[index1]
        index2 = max(num_per_class.items(), key=operator.itemgetter(1))[0]

        new_X = []
        new_y = []
        for i in range(len(y)):
            if y[i] == index1 or y[i] == index2:
                new_X.append(X[i])
                if y[i] == index1:
                    new_y.append(0)
                else:
                    new_y.append(1)
        new_y, new_X = np.asarray(new_y), np.asarray(new_X)
        return new_X, new_y
    else:
        return X, y

def get_rialto(two_classes = False):

    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'rialto', 'rialto.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'rialto', 'rialto.labels'),
        delimiter=' ')

    if two_classes:
        num_per_class = {}
        for i in range(len(y)):
            if y[i] not in num_per_class:
                num_per_class[y[i]] = 0
            else:
                num_per_class[y[i]] += 1
        import operator
        index1 = max(num_per_class.items(), key=operator.itemgetter(1))[0]
        del num_per_class[index1]
        index2 = max(num_per_class.items(), key=operator.itemgetter(1))[0]

        new_X = []
        new_y = []
        for i in range(len(y)):
            if y[i] == index1 or y[i] == index2:
                new_X.append(X[i])
                if y[i] == index1:
                    new_y.append(0)
                else:
                    new_y.append(1)
        new_y, new_X = np.asarray(new_y), np.asarray(new_X)
        return new_X, new_y
    else:
        return X, y


def get_covtype(two_classes = True):

    from scipy.io import arff
    import pandas as pd
    file = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'covType', 'covType.arff')
    a = arff.loadarff(file)
    a = pd.DataFrame(a[0])
    a = a.to_numpy().astype('float')
    X = a[:, :-1]
    y = a[:, -1].astype('int') - 1
    if two_classes:
        num_per_class = {}
        for i in range(len(y)):
            if y[i] not in num_per_class:
                num_per_class[y[i]] = 0
            else:
                num_per_class[y[i]] += 1
        import operator
        index1 = max(num_per_class.items(), key=operator.itemgetter(1))[0]
        del num_per_class[index1]
        index2 = max(num_per_class.items(), key=operator.itemgetter(1))[0]

        new_X = []
        new_y = []
        for i in range(len(y)):
            if y[i] == index1 or y[i] == index2:
                new_X.append(X[i])
                if y[i] == index1:
                    new_y.append(0)
                else:
                    new_y.append(1)
        new_y, new_X = np.asarray(new_y), np.asarray(new_X)
        return new_X, new_y
    else:
        return X, y


def get_mixeddrift():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'mixedDrift',
                     'mixedDrift.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'mixedDrift',
                     'mixedDrift.labels'),
        delimiter=' ')
    num_per_class = {}
    for i in range(len(y)):
        if y[i] not in num_per_class:
            num_per_class[y[i]] = 0
        else:
            num_per_class[y[i]] += 1
    import operator
    index1 = max(num_per_class.items(), key=operator.itemgetter(1))[0]
    del num_per_class[index1]
    index2 = max(num_per_class.items(), key=operator.itemgetter(1))[0]

    new_X = []
    new_y = []
    for i in range(len(y)):
        if y[i] == index1 or y[i] ==index2:
            new_X.append(X[i])
            if y[i] == index1:
                new_y.append(0)
            else:
                new_y.append(1)
    new_y, new_X = np.asarray(new_y), np.asarray(new_X)
    return new_X, new_y
def get_ETT():
    X = pandas.read_csv(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'realWorld', 'ETT-small',
                     'ETTm1.csv'))
    # X = X.data_frame
    return X.to_numpy()[:, 1:].astype('float32')

def get_border():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'border',
                     'border-train.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'border',
                     'border-train.labels'),
        delimiter=' ')
    return X, y

def get_COIL():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'COIL',
                     'COIL-train.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'COIL',
                     'COIL-train.labels'),
        delimiter=' ')
    return X, y

def get_MNIST():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'MNIST',
                     'mnistTrainSamples.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'MNIST',
                     'mnistTrainLabels.data'),
        delimiter=' ')
    return X, y

def get_overlap():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'overlap',
                     'overlap-train.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'overlap',
                     'overlap-train.labels'),
        delimiter=' ')
    return X, y
def get_isolet():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'isolet',
                     'isolet.data'),
        delimiter=',')
    y = X[:, -1]
    X = X[:, :-1]
    y -= 1
    return X, y

def get_gisette():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'gisette',
                     'gisette_train.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'datasets', 'gisette',
                     'gisette_train.labels'),
        delimiter=' ')
    for i in range(len(y)):
        if y[i] == -1:
            y[i] =0

    return X, y


def get_chess():
    X = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'chess',
                     'transientChessboard.data'),
        delimiter=' ')
    y = np.genfromtxt(
        os.path.join(os.path.dirname(os.path.realpath('__file__')), 'artificial', 'chess',
                     'transientChessboard.labels'),
        delimiter=' ')
    num_per_class = {}
    for i in range(len(y)):
        if y[i] not in num_per_class:
            num_per_class[y[i]] = 0
        else:
            num_per_class[y[i]] += 1
    import operator
    index1 = max(num_per_class.items(), key=operator.itemgetter(1))[0]
    del num_per_class[index1]
    index2 = max(num_per_class.items(), key=operator.itemgetter(1))[0]

    new_X = []
    new_y = []
    for i in range(len(y)):
        if y[i] == index1 or y[i] ==index2:
            new_X.append(X[i])
            if y[i] == index1:
                new_y.append(0)
            else:
                new_y.append(1)
    new_y, new_X = np.asarray(new_y), np.asarray(new_X)
    return new_X, new_y

def get_weather():
    X = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'weather', 'NEweather_data.csv'),
                      delimiter=',')
    y = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'weather', 'NEweather_class.csv'), delimiter=',')
    return X, y


def exp_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default=2, type=int, help='rank of wfa')
    parser.add_argument('--mix_n', default=2, type=int, help='num of mixture')
    parser.add_argument('--hmm_rank', default=2, type=int, help='Rank of the HMM')
    parser.add_argument('--method', default='WFA', help='method to use, either LSTM or WFA')
    parser.add_argument('--L', default=3, type=int,
                        help='length of the trajectories, WFA takes L, 2L, 2L+1, LSTM takes 2L+1')
    parser.add_argument('--N', default=100, type=int, help='number of examples to consider, WFA takes N, LSTM takes 3N')
    parser.add_argument('--xd', default=1, type=int, help='dimension of the input feature')
    parser.add_argument('--hankel_lr', default=0.01, type=float, help='hankel estimation learning rate')
    parser.add_argument('--fine_tune_lr', default=0.001, type=float, help='WFA finetune learning rate')
    parser.add_argument('--regression_lr', default=0.001, type=float, help='WFA regression learning rate')
    parser.add_argument('--LSTM_lr', default=0.001, type=float, help='LSTM learning rate')
    parser.add_argument('--hankel_epochs', default=100, type=int, help='hankel estimation epochs')
    parser.add_argument('--fine_tune_epochs', default=100, type=int, help='WFA finetune epochs')
    parser.add_argument('--regression_epochs', default=100, type=int, help='WFA regression epochs')
    parser.add_argument('--LSTM_epochs', default=100, type=int, help='WFA finetune epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--run_idx', default=0, type=int, help='index of the run')
    parser.add_argument('--seed', default=1993, type=int, help='random seed')
    parser.add_argument('--load_test_data', dest='load_data', action='store_true')
    parser.add_argument('--new_test_data', dest='load_data', action='store_false')
    parser.add_argument('--noise', default=0, type=float, help='variance of the added noise')
    parser.add_argument('--nn_transition', dest='nn_transition', action='store_true')
    parser.add_argument('--no_nn_transition', dest='nn_transition', action='store_false')
    parser.add_argument('--no_batch_norm', dest='wfa_sgd_batchnorm', action='store_false')
    parser.add_argument('--with_batch_norm', dest='wfa_sgd_batchnorm', action='store_true')
    parser.add_argument('--exp_data', default='weather', help='data to use')
    parser.add_argument('--transformer_lr', default=0.1, type = float, help = 'learning rate for transformer')
    parser.add_argument('--transformer_epochs', default=200, type = int, help = 'number of epochs for transformer')
    parser.add_argument('--nhead', default= 1, type = int, help='number of head for multihead attention')
    parser.set_defaults(load_data=True)
    return parser

def encoding(model, X):
    # X = X.reshape(1, -1)

    X = model.encoder_1(X)
    X = torch.sigmoid(X)
    X = model.encoder_2(X)
    return X
    # return X

def Fnorm(h):
    norm =  torch.norm((h), p=2)
    norm = torch.abs(norm)
    return norm

def get_parameters_dist(model, X, h):
    mu = model.mu_out(h)
    mu = mu.reshape(mu.shape[0], -1, X.shape[1])
    sig = torch.exp(model.sig_out(h))
    sig = sig.reshape(mu.shape[0], -1, X.shape[1])
    alpha = model.alpha_out(h)
    alpha = torch.softmax(alpha, dim=1)
    mix = D.Categorical(alpha)
    comp = D.Normal(mu, sig)
    comp = D.Independent(comp, 1)
    gmm = mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm.sample_n(1)

def phi(model, X, h, prediction = False, use_relu = False):

    if model.num_classes is None:
        mu = model.mu_out(h)
        if use_relu:
            mu = torch.relu(mu)
        else:
            mu = torch.relu(mu)
        mu = model.mu_out2(mu) #+ model.mu_bias

        mu = mu.reshape(mu.shape[0], -1, model.xd)

        sig = model.sig_out(h)

        # sig = sp(sig)
        # sig = torch.relu(sig)
        # sig = model.sig_out2(sig)
        # sig = torch.relu(sig)
        # sig = model.sig_out3(sig)
        sig = torch.sp(sig)#+ sp(model.sig_bias)
        # sig = torch.abs(sig)
        # print('sig', sig[0])
        #
        sig = sig.reshape(mu.shape[0], -1, model.xd)

        tmp = model.alpha_out2(h)
        # tmp = torch.relu(tmp)
        # tmp = model.alpha_out2(tmp)
        alpha = torch.softmax(tmp, dim=1)
        # print(
        #     'mu', mu[0]
        # )
        # print('alpha', alpha[0])
        # print('sig', sig[0])
        # print('h', h[0])
        # print(X.shape, mu.shape, sig.shape, alpha.shape)
        return torch_mixture_gaussian(X, mu, sig, alpha, prediction), (mu, sig, alpha)
    else:
        probs = torch.ones(model.num_classes).to(model.device)
        if not model.prob:
            probs = model.no_prob_class_out1(h)
            probs = torch.relu(probs)
            probs = model.no_prob_class_out2(probs)
            return  probs, (0, 0, 0)

        mu = model.mu_outs(h)
        mu = torch.exp(mu)
        mu = model.mu_outs2(mu)
        mu = mu.reshape(1, -1)
        mu = mu.reshape(mu.shape[0], -1, model.xd, model.num_classes)
        sig = model.sig_outs(h)
        sig = torch.exp(sig)
        sig = sig.reshape(mu.shape[0], -1, model.xd, model.num_classes)
        # if model.initial_bias is not None:
        #     for j in range(mu.shape[-1]):
        #         mu[:, :, j] = mu[:, :, j] + model.initial_bias[j]
        #     for i in range(mu.shape[-1]):
        #         mu[:, :, i] = mu[:, :, i] + model.initial_bias[i]
        alpha = model.alpha_outs(h)
        alpha = alpha.reshape(alpha.shape[0], -1, model.num_classes)


        for i in range(model.num_classes):
            mu_tmp = mu[:, :, :, i]
            sig_tmp = sig[:, :, :, i]
            alpha_tmp = alpha[:, :, i]
            alpha_tmp = torch.softmax(alpha_tmp, dim = 1)
            # print('h', h)
            # print('alpha', alpha_tmp)
            probs[i] = torch_mixture_gaussian(X, mu_tmp, sig_tmp, alpha_tmp, prediction = False)
            # probs[i] = mu[0, 0, 0]

        return probs, (mu, sig, alpha)



def phi_predict(model, X, h):
    mu = model.mu_out(h)
    mu = mu.reshape(mu.shape[0], -1, X.shape[1])
    sig = torch.exp(model.sig_out(h))
    sig = sig.reshape(mu.shape[0], -1, X.shape[1])
    tmp = model.alpha_out(h)
    alpha = torch.softmax(tmp, dim=1)
    mix = D.Categorical(alpha)
    comp = D.Normal(mu, sig)
    comp = D.Independent(comp, 1)
    # print(mu.shape, sig.shape, alpha.shape, comp)
    gmm = mixture_same_family.MixtureSameFamily(mix, comp)

    return gmm.mean, gmm.stddev


def MAPE(pred, y):
    pred = pred.reshape(y.shape)
    mape = 0.
    for i in range(len(pred)):
        if torch.any(pred [i] ==0)  or torch.any(y[i]==0):
            continue
        mape+= torch.div(torch.abs(pred[i] - y[i]), torch.abs(y[i]))
    MAPE = mape/len(pred)
    # print(MAPE)
    # print(pred[0])
    # print(y[0])
    return torch.mean(MAPE)


def torch_mixture_gaussian(X, mu, sig, alpha, prediction = False):
    index = torch.argmax(alpha)
    mix = D.Categorical(alpha)
    # sig = torch.ones(sig.shape).to('cuda:0')*0.1
    comp = D.Normal(mu, sig)
    comp = D.Independent(comp, 1)
    # print(mu.shape, sig.shape, alpha.shape, comp)
    gmm = mixture_same_family.MixtureSameFamily(mix, comp)
    # pred = torch.mean(gmm.sample_n(100), dim = 0)
    # print(gmm.sample([1])[0, 0, :], X[0], gmm.mean[0])
    if prediction:
        index = torch.argmax(alpha)
        # print(mu.shape)
        # return gmm.mean
        return gmm.sample_n(1)[0]
    else:
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

def autoregressive_regression(X, testX, L = 51, ori_X = None):
    from sklearn.linear_model import LinearRegression
    train_X = sliding_window(X, L)
    test_X = sliding_window(testX, L*10)

    train_x = train_X[:, :, :-1].swapaxes(1, 2).reshape(train_X.shape[0], -1)
    # print(train_x.shape)
    train_y = train_X[:, :, -1].reshape(train_X.shape[0], -1)
    reg = LinearRegression().fit(train_x, train_y)

    test_x = test_X[0, :, :L-1].swapaxes(0, 1).ravel()

    mape = []
    mse = []
    for i in range(test_X.shape[-1] - L):
        test_y = test_X[0, :, i+L-1].reshape(1, -1)

        # print(test_x.shape, test_X.shape)
        test_x = test_x.reshape(1, -1)
        pred_test  = reg.predict(test_x)
        test_x = test_x[:, X.shape[1]:]
        test_x = np.concatenate((test_x, pred_test), axis = 1)

        mape.append(MAPE(torch.tensor(pred_test), torch.tensor(test_y)).numpy())
        mse.append(np.mean((pred_test - test_y)**2))
        print(i, MAPE(torch.tensor(pred_test), torch.tensor(test_y)), np.mean((pred_test - test_y)**2))
        # import time
        # time.sleep(2)
    return mape, mse


if __name__ == '__main__':
    get_ETT()

