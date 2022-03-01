import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from hmmlearn import hmm
from Dataset import *
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm, insert_bias
import pickle
import sys
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
from matplotlib import pyplot as plt
from gradient_descent import *
from utils import phi, encoding,Fnorm
from Density_WFA import density_wfa
<<<<<<< Updated upstream
from torch.nn.utils.parametrizations import spectral_norm
=======
from NN_learn_transition import NN_transition_WFA
from utils import gen_hmm_parameters
torch.set_printoptions(profile="full")
>>>>>>> Stashed changes
class density_wfa_finetune(nn.Module):

    def __init__(self, density_wfa, device = device, double_pre = True, nn_transition = False, GD_linear_transition = False):
        super().__init__()

        self.device = device

        self.nade_layers = density_wfa.nade_layers
        self.mu_out = density_wfa.mu_out.requires_grad_(True)
        self.sig_out = density_wfa.sig_out.requires_grad_(True)
        self.alpha_out = density_wfa.alpha_out.requires_grad_(True)
        self.encoder_1 = density_wfa.encoder_1.requires_grad_(True)
        self.encoder_2 = density_wfa.encoder_2.requires_grad_(True)
<<<<<<< Updated upstream

        # self.scale_learnable = nn.Parameter(torch.randn(1, requires_grad=True).cuda())
        self.scale = 1.



        self.A = nn.Parameter(density_wfa.A.requires_grad_(True))
        # print(self.A)
        for i in range(self.A.shape[1]):
            # u, s, v = torch.linalg.svd(A[:, i, :])
            s = torch.linalg.svdvals(self.A[:, i, :])
            # print(s)
        # self.encoder_bn = density_wfa.encoder_bn.requires_grad_(True)
        # self.batchnorm = nn.BatchNorm1d(self.A.shape[-1])
=======
        self.GD_linear_transition = GD_linear_transition
        self.nn_transition = nn_transition
        self.A = nn.Parameter(density_wfa.A.requires_grad_(True))
        # self.batchnorm = nn.BatchNorm1d(self.A.shape[-1])
        if nn_transition or GD_linear_transition:
            self.tran_activation = density_wfa.tran_activation
>>>>>>> Stashed changes
        self.init_w = density_wfa.init_w.requires_grad_(True)
        self.scale = 1.
        self.double_pre = double_pre
<<<<<<< Updated upstream
<<<<<<< HEAD
        if self.double_pre:
            self.double()
        if self.device == 'cuda:0':
            self.cuda()
=======
        if double_pre:
            self.double().cuda()
        else:
            self.float().cuda()
>>>>>>> 42edec2714c7461aa0e4a2e76d6ca845083528a9

    def encoding(self, X):
        X = self.encoder_1(X)
        X = torch.relu(X)
        X = self.encoder_2(X)
        X =  torch.tanh(X)
        # print(X)
        return X

    def get_transition_norm(self):
        ss = []
        for i in range(self.A.shape[1]):
            u, s, v = torch.svd(self.A[:, i, :])
            ss.append(s)
        return ss

    def update_scale(self):
        max_singular_value = torch.tensor(0.).float().cuda()
        A = self.A.detach()
        for i in range(A.shape[1]):
            # u, s, v = torch.linalg.svd(A[:, i, :])
            s = torch.linalg.svdvals(A[:, i, :])
            if max_singular_value < s[0]:
                max_singular_value = s[0]
        max_singular_value = max_singular_value * A.shape[1]
        self.A.data = (A / max_singular_value)
        self.scale = self.scale * max_singular_value



    def forward(self, X):
<<<<<<< HEAD
        if double_pre:
            X = X.double()
        if device == 'cuda:0':
            X = X.cuda()
=======

        if self.double_pre:
            X = X.double().cuda()
        else:
            X = X.float().cuda()
>>>>>>> 42edec2714c7461aa0e4a2e76d6ca845083528a9

=======
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)


    # def update_scale(self):
    #     max_singular_value = torch.tensor(0.).float().cuda()
    #     A = self.A.detach()
    #     for i in range(A.shape[1]):
    #         # u, s, v = torch.linalg.svd(A[:, i, :])
    #         s = torch.linalg.svdvals(A[:, i, :])
    #         if max_singular_value < s[0]:
    #             max_singular_value = s[0]
    #     max_singular_value = A.shape[1] * max_singular_value
    #     self.A.data = (A / max_singular_value)
    #     self.scale = self.scale * max_singular_value
    #
    # def singular_value_clipping(self):
    #     A = self.A.detach()
    #     for i in range(A.shape[1]):
    #         u, s, v = torch.linalg.svd(A[:, i, :])
    #         s[s>1] = 1
    #         A[:, i, :] = u @ s @ v
    #     self.A.data = A


    def forward(self, X):
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)

>>>>>>> Stashed changes
        result = 0.
        norm = 0.
        for i in range(X.shape[2]):
            if i == 0:
                # images_repeated = images_vec.repeat(1, sequence_length)
                tmp = self.init_w.repeat(X.shape[0], 1)
                # tmp = torch.einsum("nd, ki, idj -> nkj", self.encoding(X[:, :, i]), tmp, self.A).squeeze()
            else:
<<<<<<< Updated upstream
                tmp = torch.einsum("nd, ni, idj -> nj", self.encoding(X[:, :, i-1]), tmp, self.A)


            norm += self.Fnorm(tmp)
            # print(self.scale_not_learnable, self.scale_not_learnable)
            scale = self.scale ** i
            # scaled_tmp = scale * tmp
            # print(scale)
            # print(scale*tmp)
            # print(tmp)
            tmp_result = self.phi(X[:, :, i].squeeze(), scale * tmp)

            result = result + tmp_result
            # print(result)
            # print(self.get_transition_norm())

=======
                tmp = torch.einsum("nd, ni, idj -> nj", encoding(self, X[:, :, i - 1]), tmp, self.A)
            # tmp = torch.tanh(tmp)

            # print(A, torch.einsum("nd, idj -> nij", self.encoding(X[:, :, i-1]), self.A)[2])
            if self.nn_transition or self.GD_linear_transition:
                tmp = self.tran_activation(tmp)
            # current_scale = self.scale**i
            tmp_result = phi(self, X[:, :, i].squeeze(), tmp)
            # print(tmp)
            norm += Fnorm(tmp)
            result = result + tmp_result
>>>>>>> Stashed changes
        return result, norm

    def fit(self,train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler = None, verbose = True, singular_clip_interval = 10):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
<<<<<<< Updated upstream
=======
            # if epoch % singular_clip_interval == 0:
            #     self.singular_value_clipping()
            # if epoch == epochs -1:
            #     self.update_scale()
>>>>>>> Stashed changes
            train_likehood.append(train(self, self.device, train_loader, optimizer, X = train_x, rescale=False).detach().to('cpu'))
            validation_likelihood.append(validate(self, self.device, validation_loader, X =test_x).detach().to('cpu'))
            if verbose:
                print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(train_likehood[-1],
                                                                                                     validation_likelihood[-1]))
            if scheduler is not None:
                scheduler.step(-validation_likelihood[-1])
            # for i in range(self.A.shape[1]):
            #     # u, s, v = torch.linalg.svd(A[:, i, :])
            #     print(torch.linalg.svdvals(self.A[:, i, :]))
            # print(self.scale_learnable)
            # print(self.A)
            print(self.scale)
        return train_likehood, validation_likelihood

<<<<<<< Updated upstream
    def torch_mixture_gaussian(self, X, mu, sig, alpha):
        mix = D.Categorical(alpha)
        comp = D.Normal(mu, sig)
        gmm = mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm.log_prob(X)

    def phi(self, X, h):
        # print(h.shape)
        h = torch.tanh(h)
        # print(h)
        for i in range(len(self.nade_layers)):
            h = self.nade_layers[i](h)
            h = torch.relu(h)
        mu = self.mu_out(h)
        sig = torch.exp(self.sig_out(h))
        alpha = torch.softmax(self.alpha_out(h), dim=1)



        # print('mu', mu)
        # print('x', X)
        # print('sig', sig)
        # print('alpha', alpha)
        return self.torch_mixture_gaussian(X, mu, sig, alpha)

    def eval_likelihood(self, X):
        log_likelihood, hidden_norm = self(X)
        return log_likelihood

    def Fnorm(self, h):
        return torch.norm(h, p=2)

    def negative_log_likelihood(self, X):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = torch.mean(log_likelihood)
        hidden_norm = torch.mean(hidden_norm)
        # sum_trace = 0.
        # for j in range(self.A.shape[1]):
        #     u, s, v = torch.svd(self.A[:, j, :])
        #     sum_trace += s[0] ** 2

        # print(self(X))
        # print(hidden_norm)
        return -log_likelihood


<<<<<<< HEAD
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
=======
>>>>>>> 42edec2714c7461aa0e4a2e76d6ca845083528a9

def learn_density_WFA(data, model_params, l, plot = True, out_file_name = None, load_WFA = False):
=======

    def eval_likelihood(self, X, batch = False):
        # print(X)
        log_likelihood, hidden_norm = self(X)
        if not batch:
            return torch.mean(log_likelihood)
        else:
            return log_likelihood

    def mote_carlo_phi(self, X):
        X = torch.permute(X, dims = [2, 0, 1])
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        X = encoding(self, X)
        return torch.mean(X, dim = 0)


    def lossfunc(self, X):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = torch.mean(log_likelihood)
        hidden_norm = torch.mean(hidden_norm)
        # if self.double_pre:
        #     X = X.double().cuda()
        # else:
        #     X = X.float().cuda()
        # inte_phi = self.mote_carlo_phi(X)

        # hidden_norm = torch.mean(hidden_norm)
        # sum_trace = torch.tensor(0.).cuda()
        # for j in range(self.A.shape[1]):
        #     # sum_trace += torch.sqrt(torch.trace(self.A[:, j, :]) ** 2)
        #     s = torch.linalg.svdvals(self.A[:, j, :])
        #     # print(s[0], inte_phi[j])
        #     sum_trace+= s[0] * inte_phi[j]
        # print(sum_trace)
        # print(self(X))
        return -log_likelihood



def  learn_density_WFA(data, model_params, l, plot = True, out_file_name = None, load_WFA = False, load_hankel = False, singular_clip_interval = 10, file_path = None):
>>>>>>> Stashed changes
    # data comes in a dictionary, with keys: train_l, train_2l, train_2l1, test_l, test_2l, test_2l1 indicating the length of the data
    # all data are in torch.tensor form
    # model_params is also a dictionary with keys: d (encoder output dimension), xd (original dimension of the input data), r, mixture_n, double_precision
    Ls = [l, 2*l, 2*l+1]
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
    hds = []
    lr, epochs, batch_size, double_precision = model_params['lr'], model_params['epochs'], model_params['batch_size'], model_params['double_precision']
    d, xd, r, mixture_n = model_params['d'], model_params['xd'], model_params['r'], model_params['mixture_n']
<<<<<<< Updated upstream
=======
    nn_transition = model_params['nn_transition']
    GD_linear_transition = model_params['GD_linear_transition']
    verbose = model_params['verbose']
>>>>>>> Stashed changes
    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 2}

    if not load_WFA:
<<<<<<< Updated upstream
        for k in range(len(Ls)):
            L = Ls[k]
<<<<<<< HEAD
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
=======
            train_x = data[data_label[k][0]]
            test_x = data[data_label[k][1]]
            print(data_label[k][0])
            if k == 0:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision).cuda(device)
            else:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision,
                                    previous_hd=hd).cuda(device)
>>>>>>> 42edec2714c7461aa0e4a2e76d6ca845083528a9
            train_data = Dataset(data=[train_x])
            test_data = Dataset(data=[test_x])


            train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
            test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

            optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)


            train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer,
                                               scheduler=None, verbose=True)
            if plot:
                plt.plot(train_likeli, label='train')
                plt.plot(test_likeli, label='test')
                plt.legend()
                plt.show()
            hds.append(hd)

        if out_file_name is not None:
            outfile = open(out_file_name, 'wb')
            pickle.dump(hds, outfile)
            outfile.close()

        dwfa = density_wfa(hds[0], hds[1], hds[2], double_pre=double_precision)
        # likelihood = dwfa(train_x)
        # print(likelihood)

    else:
        outfile = open(out_file_name, 'rb')
        hds = pickle.load(outfile)
        outfile.close()
        dwfa = density_wfa(hds[0], hds[1], hds[2], double_pre=double_precision)

    train_x = data['train_l']
    print('scale', dwfa.scale)
    print(torch.mean(dwfa(train_x)))

    dwfa_finetune = density_wfa_finetune(dwfa, double_pre=double_precision)


    fine_tune_lr = model_params["fine_tune_lr"]
    fine_tune_epochs = model_params["fine_tune_epochs"]
    for k in range(len(Ls)):
        print(data_label[k][0])
        train_x = data[data_label[k][0]]
        test_x = data[data_label[k][1]]
        # print(dwfa_finetune(train_x))
=======
        if load_hankel:
            outfile = open(out_file_name + 'hankels', 'rb')
            hds = pickle.load(outfile)
            outfile.close()
        else:
            for k in range(len(Ls)):
                # k = len(Ls) - k - 1
                L = Ls[k]
                train_x = data[data_label[k][0]]
                test_x = data[data_label[k][1]]
                print(data_label[k][0])
                if k == 0:
                    hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision, nn_transition=nn_transition, GD_linear_transition=GD_linear_transition).cuda(device)
                else:
                    hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision,
                                        previous_hd=hd, nn_transition=nn_transition, GD_linear_transition = GD_linear_transition).cuda(device)
                train_data = Dataset(data=[train_x])
                test_data = Dataset(data=[test_x])


                train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
                test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

                optimizer = optim.Adam(hd.parameters(), lr=lr, amsgrad=True)
                likelihood = hd(train_x)
                train_likeli, test_likeli = hd.fit(train_x, test_x, train_loader, test_loader, epochs, optimizer,
                                                   scheduler=None, verbose=verbose)
                if plot:
                    plt.plot(train_likeli, label='train')
                    plt.plot(test_likeli, label='test')
                    plt.legend()
                    plt.show()
                hds.append(hd)

        # dwfa = density_wfa(hds[0], hds[1], hds[2], double_pre=double_precision)
        if nn_transition or GD_linear_transition:
            dwfa = NN_transition_WFA(hds[0], hds[1], hds[2], double_pre=double_precision, linear_transition= GD_linear_transition)
            train_x = data[data_label[-1][0]]
            test_x = data[data_label[-1][1]]
            train_data = Dataset(data=[train_x])
            test_data = Dataset(data=[test_x])
            train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
            test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

            optimizer = optim.Adam(dwfa.parameters(), lr=lr/2, amsgrad=True)
            train_likeli, test_likeli = dwfa.fit(train_x, test_x, train_loader, test_loader, epochs*2, optimizer,
                                                 scheduler=None, verbose=verbose)
        else:
            dwfa = density_wfa(hds[0], hds[1], hds[2], double_pre=double_precision)

        if out_file_name is not None:
            outfile = open(out_file_name, 'wb')
            pickle.dump(dwfa, outfile)
            outfile.close()
            outfile = open(out_file_name+'hankels', 'wb')
            pickle.dump(hds, outfile)
            outfile.close()
    else:
        outfile = open(out_file_name, 'rb')
        dwfa = pickle.load(outfile)
        # outfile.close()
        #
        #
        # # dwfa = density_wfa(hds[0], hds[1], hds[2], double_pre=double_precision)
        #
        # dwfa = NN_transition_WFA(hds[0], hds[1], hds[2], double_pre=double_precision)
        # train_x = data[data_label[-1][0]]
        # test_x = data[data_label[-1][1]]
        # train_data = Dataset(data=[train_x])
        # test_data = Dataset(data=[test_x])
        # train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
        # test_loader = torch.utils.data.DataLoader(test_data, **generator_params)
        #
        # optimizer = optim.Adam(dwfa.parameters(), lr=lr/2, amsgrad=True)
        # train_likeli, test_likeli = dwfa.fit(train_x, test_x, train_loader, test_loader,  epochs*3, optimizer,
        #                                      scheduler=None, verbose=verbose)

    dwfa_finetune = density_wfa_finetune(dwfa, double_pre=double_precision, nn_transition=nn_transition, GD_linear_transition = GD_linear_transition)

    fine_tune_lr = model_params["fine_tune_lr"]
    fine_tune_epochs = model_params["fine_tune_epochs"]

    merged_train = []
    merged_test = []
    for k in range(len(Ls)):
        # print(data_label[k][0])
        train_x = data[data_label[k][0]]
        test_x = data[data_label[k][1]]
        merged_train.append(train_x)
        merged_test.append(test_x)
>>>>>>> Stashed changes
        train_loader = torch.utils.data.DataLoader(train_x, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_x, **generator_params)

        optimizer = optim.Adam(dwfa_finetune.parameters(), lr=fine_tune_lr, amsgrad=True)
        train_likeli, test_likeli = dwfa_finetune.fit(train_x, test_x, train_loader, test_loader, fine_tune_epochs,
                                                      optimizer, scheduler=None,
<<<<<<< Updated upstream
                                                      verbose=True)
=======
                                                      verbose=verbose, singular_clip_interval = singular_clip_interval)
>>>>>>> Stashed changes
        if plot:
            plt.plot(train_likeli, label='train')
            plt.plot(test_likeli, label='test')
            plt.legend()
            plt.show()

    return dwfa_finetune



if __name__ == '__main__':
    load = False
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
<<<<<<< Updated upstream
    N = 10000  # number of training samples
=======
    N = 100  # number of training samples
>>>>>>> Stashed changes
    d = 3  # input encoder output dimension
    xd = 1  # input dimension
    r = 3  # rank/number of mixtures
    l = 3  # length in trianning (l, 2l, 2l+1)
    Ls = [l, 2 * l, 2 * l + 1]
    mixture_n = r

    model_params = {
<<<<<<< Updated upstream
        'd':3,
        'xd': 1,
        'r': r,
        'mixture_n': mixture_n,
        'lr': 0.001,
        'epochs': 20,
        'batch_size': 256,
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.001,
        'double_precision':False
    }

    hmmmodel = hmm.GaussianHMM(n_components=3, covariance_type="full")
    hmmmodel.startprob_ = np.array([0.6, 0.3, 0.1])
    hmmmodel.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])
    hmmmodel.means_ = np.array([[0.0], [3.0], [5.0]])
    hmmmodel.covars_ = np.tile(0.1*np.identity(1), (3, 1, 1))
=======
        'd':1,
        'xd': 1,
        'r': r,
        'mixture_n': mixture_n,
        'lr': 0.01,
        'epochs': 10,
        'batch_size': 256,
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.001,
        'double_precision':False,
        'verbose': True,
        'nn_transition': False,
        'GD_linear_transition': False
    }
    hmmmodel = hmm.GaussianHMM(n_components=r, covariance_type="full")
    hmmmodel.startprob_, hmmmodel.transmat_, hmmmodel.means_, hmmmodel.covars_ = gen_hmm_parameters(r)


    # hmmmodel.startprob_ = np.array([0.6, 0.3, 0.1])
    # hmmmodel.transmat_ = np.array([[0.7, 0.2, 0.1],
    #                             [0.3, 0.5, 0.2],
    #                             [0.3, 0.3, 0.4]])
    # hmmmodel.means_ = np.array([[0.0], [3.0], [5.0]])
    # hmmmodel.covars_ = np.tile(np.identity(1), (3, 1, 1))
>>>>>>> Stashed changes

    DATA = {}

    for k in range(len(Ls)):
        L = Ls[k]
        train_x = np.zeros([N, xd, L])
        test_x = np.zeros([N, xd, L])
        for i in range(N):
            x, z = hmmmodel.sample(L)
            train_x[i, :, :] = x.reshape(xd, -1)
            x, z = hmmmodel.sample(L)
            test_x[i, :, :] = x.reshape(xd, -1)

        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()
        DATA[data_label[k][0]] = train_x
        DATA[data_label[k][1]] = test_x


    dwfa_finetune = learn_density_WFA(DATA, model_params, l)


    ls = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in ls:
        train_x = np.zeros([N, xd, 2*l])
        # print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2*l)
            train_x[i, :, :] = x.reshape(xd, -1)
        train_ground_truth = ground_truth_hmm(train_x, hmmmodel)
        train_x = torch.tensor(train_x).float()

        likelihood = dwfa_finetune.eval_likelihood(train_x)
        print("Length" + str(2 * l) + "result is:")
        print("Model output: "+str(torch.mean(likelihood)) + "Ground truth: " + str( train_ground_truth))







