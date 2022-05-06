from hmmlearn import hmm
import numpy as np

def random_startprob(r):
    tmp = np.random.rand(r)
    return tmp/np.sum(tmp)

def random_transmat(r):
    tmp = np.random.rand(r, r)
    for i in range(r):
        tmp[i] = tmp[i]/np.sum(tmp[i])
    return tmp

def random_mean(r, xd):
    tmp = np.random.rand(r, xd)*2-1
    # tmp = np.zeros((r, xd))
    return tmp

def random_covars(r, xd):
    tmp = np.random.rand(r, xd)
    return tmp

def random_weights(r, xd):
    tmp = np.random.rand(r, xd)
    for i in range(r):
        tmp[i] = tmp[i]/np.sum(tmp[i])
    return tmp


def gen_gmmhmm_data(N, xd = 2, L=3, r = 3, seed = 1993):
    np.random.seed(seed)
    hmmmodel = hmm.GaussianHMM(n_components=r, covariance_type='diag')
    hmmmodel.startprob_ = random_startprob(r)
    hmmmodel.transmat_ = random_transmat(r)
    # print(hmmmodel.transmat_.shape)
    hmmmodel.means_ = random_mean(r, xd)+10
    hmmmodel.covars_ = random_covars(r, xd)*0.1
    train_x = np.zeros([N, xd, L])
    test_x = np.zeros([N, xd, L])
    # print(train_x.shape)
    for i in range(N):
        x, z = hmmmodel.sample(L)
        train_x[i, :, :] = x.reshape(xd, L)
        x, z = hmmmodel.sample(L)
        test_x[i, :, :] = x.reshape(xd, L)
    return train_x, test_x, hmmmodel

def ground_truth(hmmmodel, X):
    log_likelihood = []
    for i in range(X.shape[0]):
        p = hmmmodel.score(X[i, :, :].transpose())
        log_likelihood.append(p)
    # print(log_likelihood)
    log_likelihood = np.asarray(log_likelihood)
    return np.mean(log_likelihood)

if __name__ == "__main__":
    train_x, test_x, hmmmodel = gen_gmmhmm_data(N = 10, xd = 2, L = 5)
    print(ground_truth(hmmmodel, train_x))


