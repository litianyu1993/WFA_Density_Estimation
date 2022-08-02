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
def get_lr(gamma, optimizer):
    return [group['lr'] * gamma
            for group in optimizer.param_groups]



def get_copy_paste(n =1000, d = 3, lag = 10, seed = 1993):

    np.random.seed(seed)
    X = np.random.normal(0, 1, size = [n, d])
    y = []
    y_neg = 0
    y_pos = 0
    for i in range(n):
        if X[i - lag][0]<= 0:
            y.append(1)
            y_pos += 1
        else:
            y.append(0)
            y_neg += 1
    y = np.asarray(y)
    print(y_pos, y_neg, y_pos/n)
    print(naive_prediction(y))
    return X, y

def naive_prediction(y):
    count = 0
    for i in range(1, len(y)):
        if y[i] == y[i-1]: count += 1
    print(count/len(y))


class incremental_HMM(nn.Module):

    def __init__(self, r, seed = 1993):
        np.random.seed(seed)
        super().__init__()
        # print('current rank is ', r)
        self.transition = torch.rand([r, r])
        self.transition = torch.softmax(self.transition, dim = 1)
        self.sig = torch.ones(r).reshape([1, r])
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
            if i % 100 == 0:
                mu+=10
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
            if i % 100 ==0:
                mu += 10
            gmm = self.torch_mixture_gaussian(mu, self.sig, h)
            current_sample = self.sample_from_gmm(gmm, 1)
            samples[:, i] = current_sample
            h = h @ self.transition
            # print(i,'sampling', mu.reshape(1, -1)@h.reshape(-1, 1))
        return samples
def get_hmm(r = 3, N = 1000):
    nshmm = incremental_HMM(r=r)
    X = nshmm.sample(N)
    X = np.asarray(X).swapaxes(0, 1)
    ground_truth_joint, ground_truth_conditionals = nshmm.score(X, stream = True)
    ground_truth_conditionals = np.asarray(ground_truth_conditionals)
    ground_truth_joint = np.asarray(ground_truth_joint)
    return X, ground_truth_conditionals, ground_truth_joint

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
    X = model.encoder_1(X)
    X = torch.relu(X)
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
            mu = torch.exp(mu)
        mu = model.mu_out2(mu)
        mu = mu.reshape(mu.shape[0], -1, model.xd)
        if model.initial_bias is not None:
            for i in range(mu.shape[-1]):
                mu[:, :, i] = mu[:, :, i] +  model.initial_bias[i]

        sig = model.sig_out(h)
        sig = torch.exp(sig)
        sig = sig.reshape(mu.shape[0], -1, model.xd)
        tmp = model.alpha_out(h)
        alpha = torch.softmax(tmp, dim=1)
        return torch_mixture_gaussian(X, mu, sig, alpha, prediction)
    else:
        probs = torch.ones(model.num_classes).to(model.device)
        for i in range(model.num_classes):

            mu = model.mu_outs[i](h)
            mu = torch.exp(mu)
            mu = model.mu_outs2[i](mu)
            mu = mu.reshape(1, -1)
            mu = mu.reshape(mu.shape[0], -1, model.xd)
            if model.initial_bias is not None:
                for j in range(mu.shape[-1]):
                    mu[:, :, j] = mu[:, :, j] + model.initial_bias[j]
            sig = model.sig_outs[i](h)
            sig = torch.exp(sig)
            sig = sig.reshape(mu.shape[0], -1, model.xd)
            tmp = model.alpha_outs[i](h)
            alpha = torch.softmax(tmp, dim=1)
            probs[i] = torch_mixture_gaussian(X, mu, sig, alpha, prediction = False)
        return probs



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

