import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from hmmlearn import hmm
from Dataset import *
from gradient_descent import train, validate
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm
import pickle
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from matplotlib import pyplot as plt
from utils import torch_mixture_gaussian, gen_hmm_parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNADE_RNN(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.input_size = parameters['input_size']
        self.RNN_hidden_size = parameters['RNN_hidden_size']
        self.RNN_num_layers = parameters['RNN_num_layers']
        self.output_size = parameters['output_size']
        self.device = parameters['device']
        self.initial_bias = parameters['initial_bias']

        print('here', self.input_size, self.RNN_hidden_size, self.RNN_num_layers)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.RNN_hidden_size, num_layers=self.RNN_num_layers, batch_first=True)
        self.tran_activation = nn.BatchNorm1d(self.RNN_hidden_size)
        # nade_hid = parameters['nade_hid']
        self.mixture_number = parameters['mixture_number']
        # self.nade_layers = nn.ModuleList()
        # self.nade_hid = [self.RNN_hidden_size, nade_hid]

        self.mu_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True)
        self.sig_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True)
        self.alpha_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number, bias=True)
        self.to(device)


    def forward(self, X, prev_state):
        X = X.to(device).float()

        result = 0.
        for i in range(X.shape[1]):
            x = X[:, i, :].reshape(X.shape[0], 1, X.shape[-1])
            state = self.tran_activation(prev_state[-1].squeeze())
            # print(state.shape)
            mu = self.mu_out(state).reshape(X.shape[0], -1, self.input_size)
            sig = torch.exp(self.sig_out(state)).reshape(X.shape[0], -1, self.input_size)
            alpha = torch.softmax(self.alpha_out(state), dim=1)
            # print(mu.shape, X.shape, sig.shape, state.shape)
            mu = mu.reshape(mu.shape[0], -1, X.shape[-1])
            if self.initial_bias is not None:
                for i in range(mu.shape[-1]):
                    mu[:, :, i] = mu[:, :, i] + self.initial_bias[i]
            sig = sig.reshape(mu.shape[0], -1, X.shape[-1])
            tmp = torch_mixture_gaussian(x.reshape(x.shape[0], -1), mu, sig, alpha)
            result += tmp
            output, (state_h, state_c) = self.lstm(x, prev_state)
            # print(state_h.shape)
            # state_h = self.tran_activation(state_h.reshape(-1, self.input_size)).reshape(1, -1, self.input_size)
            # state_h = torch.tanh(state_h)
            # state_c = torch.tanh(state_c)
            prev_state = (state_h, state_c)
            # print(output.shape, state_h.shape, state_c.shape)

            # print(X.shape, mu.shape, sig.shape, alpha.shape)
            # tmp_mu = mu[0].detach().reshape(1, self.mixture_number)
            # tmp_alpha = alpha[0].reshape(self.mixture_number)
            # print(i, tmp_mu@tmp_alpha)
            # print(i, mu[0].detach().reshape(self.mixture_number))
        return result

    def init_state(self, N):
        return (torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float(),
                torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float())

    def lossfunc(self, X):
        state_h, state_c = self.init_state(N=X.shape[0])
        log_likelihood = self(X, (state_h, state_c))
        log_likelihood = torch.mean(log_likelihood)
        # print(log_likelihood)
        return -log_likelihood

    def eval_likelihood(self, X, batch = False):
        # print(X)
        state_h, state_c = self.init_state(N=X.shape[0])
        log_likelihood = self(X, (state_h, state_c)).detach().cpu().numpy()
        likelihood = np.exp(log_likelihood)
        if not batch:
            return np.mean(log_likelihood)
        else:
            return log_likelihood



    def fit(self, train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler=None, verbose=True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
            # train(self, self.device, train_loader, optimizer, X=train_x)
            train_likehood.append(train(self, self.device, train_loader, optimizer, X=train_x))
            validation_likelihood.append(validate(self, self.device, validation_loader, X=test_x))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(
                    train_likehood[-1],
                    validation_likelihood[-1]))
            if epoch > 10 and validation_likelihood[-1] > validation_likelihood[-2]:
                count += 1
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                if count > 20: break
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])
        return train_likehood, validation_likelihood

if __name__ == '__main__':

    r = 10
    default_parameters = {
        'input_size': 1,
        'RNN_hidden_size': r,
        'RNN_num_layers': 1,
        'output_size': r,
        'mixture_number': r,
        'device': device
    }

    batch_size = 256
    lr  = 0.001
    epochs = 2

    N = 100
    xd = default_parameters['input_size']

    L = 6
    mixture_n = r

    hmmmodel = hmm.GaussianHMM(n_components=r, covariance_type="full")
    hmmmodel.startprob_, hmmmodel.transmat_, hmmmodel.means_, hmmmodel.covars_ = gen_hmm_parameters(r)
    train_x = np.zeros([N, L, xd])
    test_x = np.zeros([N, L, xd])

    for i in range(N):
        x, z = hmmmodel.sample(L)

        train_x[i, :, :] = x.reshape(L, xd)
        x, z = hmmmodel.sample(L)
        test_x[i, :, :] = x.reshape(L, xd)

    rnade_rnn =  RNADE_RNN(default_parameters)
    state_h, state_c = rnade_rnn.init_state(N=N)
    # print(train_x.shape)
    train_x = torch.from_numpy(train_x)
    # print(rnade_rnn(train_x, (state_h, state_c)).shape)

    train_data = Dataset(data=[train_x])
    test_data = Dataset(data=[test_x])

    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 0}
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

    optimizer = optim.Adam(rnade_rnn.parameters(), lr=lr, amsgrad=True)
    # likelihood = rnade_rnn(train_x)
    # print(torch.mean(torch.log(likelihood)))
    train_likeli, test_likeli = rnade_rnn.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer, scheduler=None,
                                       verbose=True)

    ls = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in ls:
        train_x = np.zeros([N, 2 * l, xd])
        # print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2 * l)
            train_x[i, :, :] = x.reshape(-1, xd)
        train_ground_truth = ground_truth_hmm(train_x.swapaxes(1, 2), hmmmodel)
        train_x = torch.tensor(train_x).float()

        likelihood = rnade_rnn.lossfunc(train_x)
        print("Length" + str(2 * l) + "result is:")
        print("Model output: " + str(torch.mean(likelihood)) + "Ground truth: " + str(train_ground_truth))









