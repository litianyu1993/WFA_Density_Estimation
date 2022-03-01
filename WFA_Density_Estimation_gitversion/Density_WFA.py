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
class density_wfa(nn.Module):

    def __init__(self, hd, h2d, h2d1, init_std = 0.1, device = device, double_pre = False):
        super().__init__()
        self.l = len(hd.core_list)
        self.d = hd.core_list[0].shape[0]
        self.r = hd.core_list[0].shape[1]

        self.device = device
        self.hankel_2l = h2d.core_list
        self.hankel_l = hd.core_list
        self.hankel_2l1 = h2d1.core_list

        self.nade_layers = hd.nade_layers
        self.mu_out = hd.mu_out
        self.sig_out = hd.sig_out
        self.alpha_out = hd.alpha_out
        self.encoder_1 = hd.encoder_1
        self.encoder_2 = hd.encoder_2

        self.A =torch.normal(0, init_std, [self.r, self.d, self.r]).float()
        self.init_w = torch.normal(0, init_std, [1, self.r]).float()
        self.spectral_learning()
        self.double_pre = double_pre
        if double_pre:
            self.double().cuda()
        else:
            self.cuda()

    def encoding(self, X):
        X = self.encoder_1(X)
        X = torch.relu(X)
        X = self.encoder_2(X)
        return torch.tanh(X)

    def forward(self, X):
        if self.double_pre:
            X = X.double().cuda()
        else:
            X = X.cuda()

        result = 0.

        for i in range(X.shape[2]):
            if i == 0:
                tmp = self.init_w
            else:
                tmp = torch.einsum("nd, ni, idj -> nj", self.encoding(X[:, :, i - 1]), tmp, self.A)
            # print(result, self.phi(X, tmp).shape)
            tmp_result = self.phi(X[:, :, i].squeeze(), tmp)
            # print(i, tmp_result)
            result = result + tmp_result
        return torch.exp(result)

    def list_to_tensor(self, core_list):
        tmp = core_list[0].detach()
        for i in range(1, len(core_list)):
            if i == 1:
                tmp = torch.tensordot(tmp, core_list[i].detach(), dims = ([1], [0]))
            else:
                tmp =torch.tensordot(tmp, core_list[i].detach(), dims = ([-1], [0]))
        return tmp


    def spectral_learning(self):
        hl = self.list_to_tensor(self.hankel_l)
        h2l = self.list_to_tensor(self.hankel_2l)
        h2l1 = self.list_to_tensor(self.hankel_2l1)

        H2l = (h2l.reshape([self.d ** self.l, self.d ** self.l * self.r]))
        H_2l1 = (h2l1.reshape([self.d ** self.l, self.d, self.d ** self.l * self.r]))
        H_l = (hl.ravel())

        U, s, V = torch.linalg.svd(H2l)
        U = U[:, :self.r]
        V = V[:self.r, :]
        s = s[:self.r]
        # print(U)

        Pinv = torch.linalg.pinv(U @ torch.diag(s))
        Sinv = torch.linalg.pinv(V)

        A = torch.tensordot(Pinv, H_2l1, dims=([1], [0]))
        A = torch.tensordot(A, Sinv, dims=([-1], [0]))
        alpha = torch.tensordot(Sinv.T, H_l.ravel().reshape(-1, 1), dims=([-1], [0]))
        self.init_w = alpha.T
        self.A = A

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




if __name__ == '__main__':
    load = False
    double_precision = True
    N = 10000
    d = 3
    xd = 1
    r = 3
    l = 3
    Ls = [2*l+1, l, 2*l]
    mixture_n = r

    batch_size = 256
    lr = 0.001
    epochs = 1

    hmmmodel = hmm.GaussianHMM(n_components=3, covariance_type="full")
    hmmmodel.startprob_ = np.array([0.6, 0.3, 0.1])
    hmmmodel.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])
    hmmmodel.means_ = np.array([[0.0], [3.0], [5.0]])
    hmmmodel.covars_ = np.tile(np.identity(1), (3, 1, 1))

    if not load:
        hds = []
        for k in range(len(Ls)):
            L = Ls[k]
            train_x = np.zeros([N, xd, L])
            test_x = np.zeros([N, xd, L])
            for i in range(N):
                x, z = hmmmodel.sample(L)
                train_x[i, :, :] = x.reshape(xd, -1)
                x, z = hmmmodel.sample(L)
                test_x[i, :, :] = x.reshape(xd, -1)
            test_ground_truth = ground_truth_hmm(test_x, hmmmodel)
            train_ground_truth = ground_truth_hmm(train_x, hmmmodel)

            print(test_ground_truth, train_ground_truth)

            train_x = (torch.tensor(train_x).float())
            test_x = (torch.tensor(test_x).float())

            if k == 0:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L = L, double_pre=double_precision).cuda(device)
            else:
                hd = hankel_density(d, xd, r, mixture_number=mixture_n, L = L, double_pre=double_precision, previous_hd=hd).cuda(device)
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

    ls = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in ls:
        train_x = np.zeros([N, xd, 2*l])
        print(2*l)
        for i in range(N):
            x, z = hmmmodel.sample(2*l)
            train_x[i, :, :] = x.reshape(xd, -1)
        train_ground_truth = ground_truth_hmm(train_x, hmmmodel)
        train_x = torch.tensor(train_x).float()
        dwfa = density_wfa(hds[1], hds[2], hds[0], double_pre=double_precision)
        likelihood = dwfa(train_x)
        print(torch.mean(torch.log(likelihood)), train_ground_truth)







