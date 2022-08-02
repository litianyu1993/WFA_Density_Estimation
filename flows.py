import math
import types

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch import optim

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset

import tqdm
import matplotlib.pyplot as plt


def potential_fn(dataset):
    # NF paper table 1 energy functions
    w1 = lambda z: torch.sin(2 * math.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
    w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    if dataset == 'u1':
        return lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4) ** 2 - \
                         torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                                   torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-10)

    elif dataset == 'u2':
        return lambda z: 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2

    elif dataset == 'u3':
        return lambda z: - torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2) + \
                                     torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) + 1e-10)

    elif dataset == 'u4':
        return lambda z: - torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2) + \
                                     torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) + 1e-10)

    else:
        raise RuntimeError('Invalid potential name to sample from.')


def sample_2d_data(dataset, n_samples):
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1 / math.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1 * z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08 * torch.ones_like(x))

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')


# --------------------
# Model components
# --------------------

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, data_dim):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data_dim = data_dim

        # Notation:
        # BNAF weight calculation for (eq 8): W = g(W) * M_d + W * M_o
        #   where W is block lower triangular so model is autoregressive,
        #         g = exp function; M_d is block diagonal mask; M_o is block off-diagonal mask.
        # Weight Normalization (Salimans & Kingma, eq 2): w = g * v / ||v||
        #   where g is scalar, v is k-dim vector, ||v|| is Euclidean norm
        # ------
        # Here: pre-weight norm matrix is v; then: v = exp(weight) * mask_d + weight * mask_o
        #       weight-norm scalar is g: out_features dimensional vector (here logg is used instead to avoid taking logs in the logdet calc.
        #       then weight-normed weight matrix is w = g * v / ||v||
        #
        #       log det jacobian of block lower triangular is taking block diagonal mask of
        #           log(g*v/||v||) = log(g) + log(v) - log(||v||)
        #                          = log(g) + weight - log(||v||) since v = exp(weight) * mask_d + weight * mask_o

        weight = torch.zeros(out_features, in_features)
        mask_d = torch.zeros_like(weight)
        mask_o = torch.zeros_like(weight)
        for i in range(data_dim):
            # select block slices
            h = slice(i * out_features // data_dim, (i + 1) * out_features // data_dim)
            w = slice(i * in_features // data_dim, (i + 1) * in_features // data_dim)
            w_row = slice(0, (i + 1) * in_features // data_dim)
            # initialize block-lower-triangular weight and construct block diagonal mask_d and lower triangular mask_o
            nn.init.kaiming_uniform_(weight[h, w_row], a=math.sqrt(5))  # default nn.Linear weight init only block-wise
            mask_d[h, w] = 1
            mask_o[h, w_row] = 1

        mask_o = mask_o - mask_d  # remove diagonal so mask_o is lower triangular 1-off the diagonal

        self.weight = nn.Parameter(weight)  # pre-mask, pre-weight-norm
        self.logg = nn.Parameter(torch.rand(out_features, 1).log())  # weight-norm parameter
        self.bias = nn.Parameter(nn.init.uniform_(torch.rand(out_features), -1 / math.sqrt(in_features),
                                                  1 / math.sqrt(in_features)))  # default nn.Linear bias init
        self.register_buffer('mask_d', mask_d)
        self.register_buffer('mask_o', mask_o)

    def forward(self, x, sum_logdets):
        # 1. compute BNAF masked weight eq 8
        v = self.weight.exp() * self.mask_d + self.weight * self.mask_o
        # 2. weight normalization
        v_norm = v.norm(p=2, dim=1, keepdim=True)
        w = self.logg.exp() * v / v_norm
        # 3. compute output and logdet of the layer
        out = F.linear(x, w, self.bias)
        logdet = self.logg + self.weight - 0.5 * v_norm.pow(2).log()
        logdet = logdet[self.mask_d.byte()]
        # print(out.shape[1], self.data_dim, x.shape[1])
        logdet = logdet.view(1, self.data_dim, out.shape[1] // self.data_dim, x.shape[1] // self.data_dim) \
            .expand(x.shape[0], -1, -1, -1)  # output (B, data_dim, out_dim // data_dim, in_dim // data_dim)

        # 4. sum with sum_logdets from layers before (BNAF section 3.3)
        # Compute log det jacobian of the flow (eq 9, 10, 11) using log-matrix multiplication of the different layers.
        # Specifically for two successive MaskedLinear layers A -> B with logdets A and B of shapes
        #  logdet A is (B, data_dim, outA_dim, inA_dim)
        #  logdet B is (B, data_dim, outB_dim, inB_dim) where outA_dim = inB_dim
        #
        #  Note -- in the first layer, inA_dim = in_features//data_dim = 1 since in_features == data_dim.
        #            thus logdet A is (B, data_dim, outA_dim, 1)
        #
        #  Then:
        #  logsumexp(A.transpose(2,3) + B) = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, inB_dim) , dim=-1)
        #                                  = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, outA_dim), dim=-1)
        #                                  = logsumexp( (B, data_dim, outB_dim, outA_dim), dim=-1) where dim2 of tensor1 is broadcasted
        #                                  = (B, data_dim, outB_dim, 1)

        sum_logdets = torch.logsumexp(sum_logdets.transpose(2, 3) + logdet, dim=-1, keepdim=True)

        return out, sum_logdets

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sum_logdets):
        # derivation of logdet:
        # d/dx tanh = 1 / cosh^2; cosh = (1 + exp(-2x)) / (2*exp(-x))
        # log d/dx tanh = - 2 * log cosh = -2 * (x - log 2 + log(1 + exp(-2x)))
        logdet = -2 * (x - math.log(2) + F.softplus(-2 * x))
        sum_logdets = sum_logdets + logdet.view_as(sum_logdets)
        return x.tanh(), sum_logdets


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x):
        sum_logdets = torch.zeros(1, x.shape[1], 1, 1, device=x.device)
        for module in self:
            x, sum_logdets = module(x, sum_logdets)
        return x, sum_logdets.squeeze()


# --------------------
# Model
# --------------------

class BNAF(nn.Module):
    def __init__(self, n_input, n_layers, n_hidden, seed = 1993):
        super().__init__()
        torch.manual_seed(seed)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(n_input))
        self.register_buffer('base_dist_var', torch.ones(n_input))

        # construct model
        modules = []
        modules += [MaskedLinear(n_input, n_hidden, n_input), Tanh()]
        for _ in range(n_layers):
            modules += [MaskedLinear(n_hidden, n_hidden, n_input), Tanh()]
        modules += [MaskedLinear(n_hidden, n_input, n_input)]
        self.net = FlowSequential(*modules)

        # TODO --   add permutation
        #           add residual gate
        #           add stack of flows

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        return self.net(x)

    def log_probability(self, x):
        z, logdet = self.forward(x)
        return torch.sum(self.base_dist.log_prob(z) + logdet, dim=1)


class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

        self.scale = nn.Parameter(torch.Tensor(self.dim))

        self.t_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

    def forward(self, x):
        x_mask = x * self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)

        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac


class RealNVP(nn.Module):
    def __init__(self, n_input, n_layers, n_hidden):
        super(RealNVP, self).__init__()

        # masks = torch.cat([torch.stack([
        #         torch.arange(1, n_input+1).float() % 2,
        #         torch.arange(0, n_input).float() % 2
        #         ]
        #         ) for i in range(n_layers//2)], 0)
        masks = torch.FloatTensor(np.mod(np.arange(n_input).reshape(-1, 1) + np.arange(n_input), 2).astype(np.float32))
        config = 1
        masks = torch.stack([masks[0] if config == 0 else masks[1] for i in range(n_layers)])

        self.dim = len(masks[0])
        self.hidden_size = n_hidden

        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNode(mask, self.hidden_size) for mask in self.masks])

        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0])
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        log_prob += self.distribution.log_prob(x)

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)

        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac

        return x, log_prob

    def sample_each_step(self, num_samples):
        samples = []

        x = self.distribution.sample((num_samples,))
        samples.append(x.detach().numpy())

        for layer in self.layers:
            x, _ = layer.forward(x)
            samples.append(x.detach().numpy())

        return samples


def train(model, data, epochs=100, batch_size=512):
    train_data = data['train']
    test_data = data['test']
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
        epoch_loss = 0
        for epoch in tepoch:
            for batch_index, training_sample in enumerate(train_loader):
                training_sample = training_sample.view(training_sample.shape[0], -1).float()
                log_prob = model.log_probability(training_sample)
                loss = - log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            epoch_loss /= len(train_loader)
            losses.append(np.copy(epoch_loss.detach().numpy()))
            tepoch.set_postfix(loss=epoch_loss.detach().numpy())
    # print(test_data.X.shape)
    test_data = test_data.X.view(test_data.X.shape[0], -1).float()
    # print(-model.log_probability(test_data))
    test_loss = -model.log_probability(test_data).mean().detach().numpy()
    # train_data = torch.tensor(train_data.X)
    # print(test_loss, -model.log_probability(train_data.view(train_data.shape[0], -1).float()).mean().detach().numpy())
    return model, losses, test_loss


def plot_density(model, true_dist=None, num_samples=100, mesh_size=4.):
    x_mesh, y_mesh = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=num_samples),
                                 np.linspace(- mesh_size, mesh_size, num=num_samples))

    cords = np.stack((x_mesh, y_mesh), axis=2)
    cords_reshape = cords.reshape([-1, 2])
    log_prob = np.zeros((num_samples ** 2))

    for i in range(0, num_samples ** 2, num_samples):
        data = torch.from_numpy(cords_reshape[i:i + num_samples, :]).float()
        log_prob[i:i + num_samples] = model.log_probability(data).cpu().detach().numpy()

    plt.scatter(cords_reshape[:, 0], cords_reshape[:, 1], c=np.exp(log_prob))
    plt.colorbar()
    if true_dist is not None:
        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', alpha=.05)
    plt.show()


def make_art_gaussian(n_gaussians=3, n_samples=1000):
    radius = 2.5
    angles = np.linspace(0, 2 * np.pi, n_gaussians, endpoint=False)

    cov = np.array([[.1, 0], [0, .1]])
    results = []

    for angle in angles:
        results.append(
            np.random.multivariate_normal(radius * np.array([np.cos(angle), np.sin(angle)]), cov,
                                          int(n_samples / 3) + 1))

    return np.random.permutation(np.concatenate(results))


class FlowDataset(Dataset):
    def __init__(self, dataset_type, num_samples=1000, seed=0, **kwargs):
        """
        Dataset used to load different artificial datasets to train normalizing flows on.
        Args:
        dataset_type (str): Choose type from: MultiVariateNormal, Moons, Circles or MultipleGaussians
        num_samples (int): Number of samples to draw.
        seed (int): Random seed.
        kwargs: Specific parameters for the different distributions.
        """
        np.random.seed(seed)
        if dataset_type == 'MultiVariateNormal':
            mean = kwargs.pop('mean', [0, 3, 4, 5, 6, 7, 8, 9])
            cov = kwargs.pop('mean', np.diag([.1, .1, .1, .1, .1, .1, .1, .1]))
            self.data = np.random.multivariate_normal(mean, cov, num_samples)
        elif dataset_type == 'Moons':
            noise = kwargs.pop('noise', .1)
            self.data = make_moons(num_samples, noise=noise, random_state=seed, shuffle=True)[0]
        elif dataset_type == 'Circles':
            factor = kwargs.pop('factor', .5)
            noise = kwargs.pop('noise', .05)
            self.data = make_circles(num_samples, noise=noise, factor=factor, random_state=seed, shuffle=True)[0]
        elif dataset_type == 'MultipleGaussians':
            num_gaussians = kwargs.pop('num_gaussians', 3)
            self.data = make_art_gaussian(num_gaussians, num_samples)
        else:
            raise NotImplementedError

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).type(torch.FloatTensor)

class TorchDataset(Dataset):
    def __init__(self, X, **kwargs):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return torch.from_numpy(self.X[index]).type(torch.FloatTensor)
        return torch.from_numpy(self.X[index]).type(torch.FloatTensor)


class NumPyDataset(Dataset):
    def __init__(self, X, **kwargs):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return torch.from_numpy(self.X[index]).type(torch.FloatTensor)
        return self.X[index]


if __name__ == '__main__':
    n_input = 8
    n_layers = 4
    n_hidden = 256

    data = FlowDataset('MultiVariateNormal', num_gaussians=5)
    print(data.data.shape, data.num_samples)
    # NVP_model = RealNVP(n_input=n_input, n_layers=n_layers, n_hidden=n_hidden)
    bnaf_model = BNAF(n_input=n_input, n_layers=n_layers, n_hidden=n_hidden)
    flow_model, loss, test_loss = train(bnaf_model, {'train': data, 'test': data[:128]}, epochs=100)

    plot_density(flow_model)