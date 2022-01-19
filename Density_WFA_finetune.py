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
from torch.nn.utils.parametrizations import spectral_norm
from flows import FlowDensityEstimator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.init_w = density_wfa.init_w.requires_grad_(True)
        self.double_pre = double_pre
        if double_pre:
            self.double()
        else:
            self.float()
        if device == 'cuda':
            self.cuda()

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
        if self.double_pre:
            X = X.double()
        else:
            X = X.float()
        if device == 'cuda':
            X = X.cuda()

        result = 0.
        norm = 0.
        for i in range(X.shape[2]):
            if i == 0:
                tmp = self.init_w
            else:
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

        return result, norm

    def fit(self,train_x, test_x, train_loader, validation_loader, epochs, optimizer, scheduler = None, verbose = True):
        train_likehood = []
        validation_likelihood = []
        count = 0
        for epoch in range(epochs):
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
        # for j in range(model.A.shape[1]):
        #     u, s, v = torch.svd(model.A[:, j, :])
        #     sum_trace += s[0] ** 2

        return -log_likelihood #+ sum_trace



def learn_density_WFA(data, model_params, l, plot = True, out_file_name = None, load_WFA = False):
    # data comes in a dictionary, with keys: train_l, train_2l, train_2l1, test_l, test_2l, test_2l1 indicating the length of the data
    # all data are in torch.tensor form
    # model_params is also a dictionary with keys: d (encoder output dimension), xd (original dimension of the input data), r, mixture_n, double_precision
    Ls = [l, 2*l, 2*l+1]
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
    hds = []
    lr, epochs, batch_size, double_precision = model_params['lr'], model_params['epochs'], model_params['batch_size'], model_params['double_precision']
    d, xd, r, mixture_n = model_params['d'], model_params['xd'], model_params['r'], model_params['mixture_n']
    generator_params = {'batch_size': batch_size,
                        'shuffle': False,
                        'num_workers': 2}

    if not load_WFA:
        for k in range(len(Ls)):
            L = Ls[k]
            train_x = data[data_label[k][0]]
            test_x = data[data_label[k][1]]
            
            if k == 0:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision).to(device)
            else:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=L, double_pre=double_precision,
                                    previous_hd=hd).to(device)
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
        train_loader = torch.utils.data.DataLoader(train_x, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_x, **generator_params)

        optimizer = optim.Adam(dwfa_finetune.parameters(), lr=fine_tune_lr, amsgrad=True)
        train_likeli, test_likeli = dwfa_finetune.fit(train_x, test_x, train_loader, test_loader, fine_tune_epochs,
                                                      optimizer, scheduler=None,
                                                      verbose=True)
        if plot:
            plt.plot(train_likeli, label='train')
            plt.plot(test_likeli, label='test')
            plt.legend()
            plt.show()

    return dwfa_finetune



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


    dwfa_finetune = learn_density_WFA(DATA, model_params, l, plot=False)


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
        flow = FlowDensityEstimator('realnvp', num_inputs=train_x.shape[-1], num_hidden=64, num_blocks=5, num_cond_inputs=None, act='relu', device='cpu')
        train_lik, test_lik = flow.train({'train': train_x, 'test': train_x }, batch_size=train_x.shape[0], epochs=50)
        print("Length" + str(2 * l) + "result is:")
        print("Model output: "+str(torch.mean(likelihood).detach().cpu().item()) + " RealNVP: " + str(-train_lik) + " Ground truth: " + str( train_ground_truth))







