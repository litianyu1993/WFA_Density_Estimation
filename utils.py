import torch
from torch import nn
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
import numpy as np
from hmmlearn import hmm
import pickle
import argparse

def sliding_window(X, window_size = 5):
    final_data = []
    for j in range(len(X) - 1 - window_size):
        tmp = X[j:j+window_size]
        final_data.append(tmp)
    return np.asarray(final_data).swapaxes(1, 2)

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
    # X = model.encoder_1(X)
    # X = torch.relu(X)
    # X = model.encoder_2(X)
    # return torch.tanh(X)
    return X

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

def phi(model, X, h, prediction = False):
    # print(h.shape)
    # h = torch.tanh(h)
    # print('h1', h)
    # for i in range(len(model.nade_layers)):
    #     h = model.nade_layers[i](h)
    #     h = torch.tanh(h)
    # print(h.shape)
    # print(h.shape, model.mu_out.shape, model.mu_out_bias.shape)
    # mu = torch.einsum('ij, jkl -> ikl', h, model.mu_out)+ model.mu_out_bias
    # print('1', h, torch.any(torch.isnan(h)))
    mu = model.mu_out(h)
    # mu = torch.tanh(mu)
    # mu = model.mu_out2(mu)
    mu = mu.reshape(mu.shape[0], -1, X.shape[1])

    if model.initial_bias is not None:
        for i in range(mu.shape[-1]):
            mu[:, :, i] =  mu[:, :, i] + model.initial_bias[i]
    # sig =  torch.einsum('ij, jkl -> ikl', h, model.sig_out)+ model.sig_out_bias
    # sig = torch.exp(sig)
    sig = model.sig_out(h)
    # sig = torch.tanh(sig)
    # sig = model.sig_out2(sig)
    sig = torch.exp(sig)
    sig = sig.reshape(mu.shape[0], -1, X.shape[1])
    # print(torch.mean(sig))
    # print(h)
    # h = h - torch.max(h)
    # print('h', h)
    # print('alpha', model.alpha_out(h))
    # print(h.shape)
    # print('2', h, torch.any(torch.isnan(h)))
    tmp = model.alpha_out(h)
    # tmp = torch.tanh(tmp)
    # tmp = model.alpha_out2(tmp)
    alpha = torch.softmax(tmp, dim=1)
    return torch_mixture_gaussian(X, mu, sig, alpha, prediction)

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

    mix = D.Categorical(alpha) 
    comp = D.Normal(mu, sig)
    comp = D.Independent(comp, 1)
    # print(mu.shape, sig.shape, alpha.shape, comp)
    gmm = mixture_same_family.MixtureSameFamily(mix, comp)
    # print(gmm.sample([1])[0, 0, :], X[0], gmm.mean[0])
    if prediction:
        return gmm.mean, gmm.stddev
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

def autoregressive_regression(X, testX, L = 50):
    from sklearn.linear_model import LinearRegression
    train_X = sliding_window(X, L)
    test_X = sliding_window(testX, L*10)

    train_x = train_X[:, :, :-1].swapaxes(1, 2).reshape(train_X.shape[0], -1)
    # print(train_x.shape)
    train_y = train_X[:, :, -1].reshape(train_X.shape[0], -1)
    reg = LinearRegression().fit(train_x, train_y)

    test_x = test_X[:, :, :L-1].swapaxes(1, 2).reshape(test_X.shape[0], -1)
    for i in range(test_X.shape[-1] - L):
        test_y = test_X[:, :, i+L-1].reshape(test_X.shape[0], -1)

        # print(test_x.shape, test_X.shape)
        pred_test  = reg.predict(test_x)
        test_x = test_x[:, X.shape[1]:]
        test_x = np.concatenate((test_x, pred_test), axis = 1)

        print(i, MAPE(torch.tensor(pred_test), torch.tensor(test_y)), np.mean((pred_test - test_y)**2))
        # import time
        # time.sleep(2)
    return MAPE(torch.tensor(pred_test), torch.tensor(test_y)), np.mean((pred_test - test_y)**2)


if __name__ == '__main__':
    rs = [5, 10, 20, 40]
    for r in rs:
        hmmmodel = hmm.GaussianHMM(n_components=r, covariance_type="full")
        hmmmodel.startprob_, hmmmodel.transmat_, hmmmodel.means_, hmmmodel.covars_ = gen_hmm_parameters(r)
        with open('rank_'+str(r), 'wb') as f:
            pickle.dump(hmmmodel, f)


