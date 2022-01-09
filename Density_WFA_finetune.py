import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from hmmlearn import hmm
from Dataset import *
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm, insert_bias
import pickle
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from matplotlib import pyplot as plt
from gradient_descent import *
from Density_WFA import density_wfa
class density_wfa_finetune(nn.Module):

    def __init__(self, density_wfa, device = device, double_pre = True):
        super().__init__()

        self.device = device

        self.nade_layers = density_wfa.nade_layers
        self.mu_out = density_wfa.mu_out.requires_grad_(True)
        self.sig_out = density_wfa.sig_out.requires_grad_(True)
        self.alpha_out = density_wfa.alpha_out.requires_grad_(True)
        self.encoder_1 = density_wfa.encoder_1.requires_grad_(True)
        self.encoder_2 = density_wfa.encoder_2.requires_grad_(True)

        self.A = density_wfa.A.requires_grad_(True)
        self.init_w = density_wfa.init_w.requires_grad_(True)
        self.double_pre = double_pre
        if self.double_pre:
            self.double()
        if self.device == 'cuda:0':
            self.cuda()

    def encoding(self, X):
        X = self.encoder_1(X)
        X = torch.relu(X)
        X = self.encoder_2(X)
        return torch.tanh(X)

    def forward(self, X):
        if double_pre:
            X = X.double()
        if device == 'cuda:0':
            X = X.cuda()

        result = 0.

        for i in range(X.shape[2]):
            if i == 0:
                tmp = self.init_w
            else:
                tmp = torch.einsum("nd, ni, idj -> nj", self.encoding(X[:, :, i - 1]), tmp, self.A)
            tmp_result = self.phi(X[:, :, i].squeeze(), tmp)
            result = result + tmp_result
        return torch.exp(result)

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

        return train_likehood, validation_likelihood

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
        alpha = torch.softmax(self.alpha_out(h), dim=1)
        return self.torch_mixture_gaussian(X, mu, sig, alpha)
    def eval_likelihood(self, X):
        likelihood = self(X) #sum_to_one

        return torch.mean(torch.log(likelihood))

    def negative_log_likelihood(self, X):
        log_likelihood = torch.mean(torch.log(self(X)))
        sum_trace = 0.
        for j in range(self.A.shape[1]):
            sum_trace += torch.sqrt(torch.trace(self.A[:, j, :]) ** 2)
        # print(self(X))
        return -log_likelihood+sum_trace



if __name__ == '__main__':
    load = False
    double_precision = True
    N = 10000 #number of training samples
    d = 3 #input encoder output dimension
    xd = 1 #input dimension
    r = 3 #rank/number of mixtures
    l = 3 #length in training (l, 2l, 2l+1)
    Ls = [2*l+1, l, 2*l]
    mixture_n = r

    batch_size = 256
    lr = 0.001
    epochs = 100

    n_hmms = 4
    n_states = 3
    hmm_list = []
    for i in range(n_hmms):
        hmmmodel = hmm.GaussianHMM(n_components=3, covariance_type="full")
        startprob = np.random.uniform(size=(n_states,))
        startprob = startprob / startprob.sum()
        transmat = np.random.uniform(size=(n_states, n_states))
        transmat = transmat / transmat.sum(1)[:,None]
        means = np.random.uniform(size=(n_states,1))
        hmmmodel.startprob_ = startprob
        hmmmodel.transmat_ = transmat
        hmmmodel.means_ = means
        # hmmmodel.startprob_ = np.array([0.6, 0.3, 0.1])
        # hmmmodel.transmat_ = np.array([[0.7, 0.2, 0.1],
        #                             [0.3, 0.5, 0.2],
        #                             [0.3, 0.3, 0.4]])
        # hmmmodel.means_ = np.array([[0.0], [3.0], [5.0]])
        hmmmodel.covars_ = np.tile(np.identity(1), (n_states, 1, 1))
        hmm_list.append(hmmmodel)


    if not load:
        hds = []
        for k in range(len(Ls)):
            L = Ls[k]
            train_x = np.zeros([N, xd, L])
            test_x = np.zeros([N, xd, L])
            test_ground_truth_list = []
            train_ground_truth_list = []
            for h in range(n_hmms):
                for i in range(N//n_hmms):
                    x, z = hmm_list[h].sample(L)
                    train_x[i, :, :] = x.reshape(xd, -1)
                    x, z = hmm_list[h].sample(L)
                    test_x[i, :, :] = x.reshape(xd, -1)

                test_ground_truth = ground_truth_hmm(test_x, hmm_list[h])
                train_ground_truth = ground_truth_hmm(train_x, hmm_list[h])
                test_ground_truth_list.append(test_ground_truth)
                train_ground_truth_list.append(train_ground_truth)

            print(test_ground_truth_list, train_ground_truth_list)

            train_x = torch.tensor(train_x).float()
            test_x = torch.tensor(test_x).float()

            # else:
            #     final_train = torch.concat([final_train, train_x])
            #     final_test = torch.concat([final_test, test_x])

            if k == 0:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L = L, double_pre=double_precision).to(device)
            else:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L = L, double_pre=double_precision, previous_hd=hd).to(device)
            train_data = Dataset(data=[train_x])
            test_data = Dataset(data=[test_x])

            generator_params = {'batch_size': batch_size,
                                'shuffle': False,
                                'num_workers': 2}
            train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
            test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

            optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)
            likelihood =hd(train_x)
            train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer, scheduler=None, verbose=True)
            hds.append(hd)
            plt.plot(train_likeli, label='train')
            plt.plot(test_likeli, label='test')
            plt.plot(test_ground_truth * np.ones(len(test_likeli)), label='test truth')
            plt.plot(train_ground_truth * np.ones(len(train_likeli)), label='train truth')
            plt.legend()
            plt.show()

        outfile = open('density_hd', 'wb')
        pickle.dump(hds, outfile)
        outfile.close()
    else:
        outfile = open('density_hd', 'rb')
        hds = pickle.load(outfile)
        outfile.close()


    dwfa = density_wfa(hds[1], hds[2], hds[0], double_pre=double_precision)
    dwfa_finetune = density_wfa_finetune(dwfa)
    # for name, param in dwfa_finetune.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    ls = [3, 6, 7]
    for l in ls:
        final_train = np.zeros([N, xd, l])
        final_test = np.zeros([N, xd, l])
        for i in range(N):
            x, z = hmmmodel.sample(l)
            final_train[i, :, :] = x.reshape(xd, -1)
            x, z = hmmmodel.sample(l)
            final_test[i, :, :] = x.reshape(xd, -1)
        train_ground_truth = ground_truth_hmm(final_train, hmmmodel)
        test_ground_truth = ground_truth_hmm(final_test, hmmmodel)
        final_train = torch.tensor(final_train).float()
        final_test = torch.tensor(final_test).float()
        train_loader = Dataset(data=[final_train])
        test_loader = Dataset(data=[final_test])

        generator_params = {'batch_size': batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        train_loader = torch.utils.data.DataLoader(final_train, **generator_params)
        test_loader = torch.utils.data.DataLoader(final_test, **generator_params)

        optimizer = optim.Adam(dwfa_finetune.parameters(), lr=lr, amsgrad=True)
        train_likeli, test_likeli = dwfa_finetune.fit(final_train, final_test, train_loader, test_loader, epochs, optimizer, scheduler=None,
                                           verbose=True)
        plt.plot(train_likeli, label='train')
        plt.plot(test_likeli, label='test')
        plt.plot(test_ground_truth * np.ones(len(test_likeli)), label='test truth')
        plt.plot(train_ground_truth * np.ones(len(train_likeli)), label='train truth')
        plt.legend()
        plt.show()


    ls = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in ls:
        train_x = np.zeros([N, xd, 2*l])
        print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2*l)
            train_x[i, :, :] = x.reshape(xd, -1)
        train_ground_truth = ground_truth_hmm(train_x, hmmmodel)
        train_x = torch.tensor(train_x).float()

        likelihood = dwfa_finetune(train_x)
        print("Length" + str(2 * l) + "result is:")
        print("Model output: "+str(torch.mean(torch.log(likelihood))) + "Ground truth: " + str( train_ground_truth))







