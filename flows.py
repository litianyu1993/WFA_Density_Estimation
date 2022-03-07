import math
import types

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset

import tqdm
import matplotlib.pyplot as plt

class FlowDensityEstimator(nn.Module):
    def __init__(self, flow, num_inputs=1, num_hidden=64, num_blocks=1, num_cond_inputs=None, lr=3e-4, act='relu', device='cpu'):
        super(FlowDensityEstimator, self).__init__()
        self.device = device
        self.flow = flow
        assert flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
        modules = []
        if flow == 'glow':
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.to(device).float()

            for _ in range(num_blocks):
                modules += [
                    BatchNormFlow(num_inputs),
                    LUInvertibleMM(num_inputs),
                    CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu')
                ]
                mask = 1 - mask
        elif flow == 'realnvp':
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.to(device).float()

            for _ in range(num_blocks):
                modules += [
                    CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu'),
                    BatchNormFlow(num_inputs)
                ]
                mask = 1 - mask
        elif flow == 'maf':
            for _ in range(num_blocks):
                modules += [
                    MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                    BatchNormFlow(num_inputs),
                    Reverse(num_inputs)
                ]
        elif flow == 'maf-split':
            for _ in range(num_blocks):
                modules += [
                    MADESplit(num_inputs, num_hidden, num_cond_inputs,
                                s_act='tanh', t_act='relu'),
                    BatchNormFlow(num_inputs),
                    Reverse(num_inputs)
                ]
        elif flow == 'maf-split-glow':
            for _ in range(num_blocks):
                modules += [
                    MADESplit(num_inputs, num_hidden, num_cond_inputs,
                                s_act='tanh', t_act='relu'),
                    BatchNormFlow(num_inputs),
                    InvertibleMM(num_inputs)
                ]

        self.model = FlowSequential(*modules)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)

    def train(self, data, batch_size=128, epochs=10):
        self.model.train()
        # n_samples x T x n_d
        train_data = data['train'].float()
        test_data = data['test'].float()
        
        for e in range(epochs):
            train_loss = 0.
            n_batches = train_data.shape[0]//batch_size
            idx = torch.rand(n_batches * batch_size).argsort().view(n_batches, batch_size)
            
            for batch_idx in idx:
                batch = train_data[batch_idx]
                batch = batch.to(self.device)
                cond_data = None
                self.optimizer.zero_grad()
                loss = -self.model.log_probs(batch, cond_data).mean()
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / n_batches
            validation_loss = -self.model.log_probs(test_data, cond_data).mean()
            # print('Epoch: %d Train Likelihood: %f Validate Likelihood: %f'%(e, -train_loss, -validation_loss))
        return train_loss, validation_loss

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class SimpleAffine(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.zeros(self.dim))  # log_scale
        self.b = nn.Parameter(torch.zeros(self.dim))  # shift

    def forward(self, x):
        y = torch.exp(self.a) * x + self.b

        det_jac = torch.exp(self.a.sum())
        log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jac)

        return y, log_det_jac

    def inverse(self, y):
        x = (y - self.b) / torch.exp(self.a)

        det_jac = 1 / torch.exp(self.a.sum())
        inv_log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jac)

        return x, inv_log_det_jac


class StackSimpleAffine(nn.Module):
    def __init__(self, transforms, dim=2):
        super().__init__()
        self.dim = dim
        self.transforms = nn.ModuleList(transforms)
        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0])
        for transform in reversed(self.transforms):
            x, inv_log_det_jac = transform.inverse(x)
            log_prob += inv_log_det_jac

        log_prob += self.distribution.log_prob(x)

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)

        for transform in self.transforms:
            x, log_det_jac = transform.forward(x)
            log_prob += log_det_jac

        return x, log_prob


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
        x_mask = x*self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x*torch.exp(s) + t)

        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1-self.mask)*(y - t)*torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac


class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size):
        super(RealNVP, self).__init__()

        self.dim = len(masks[0])
        self.hidden_size = hidden_size

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

def train(model, data, epochs = 100, batch_size = 64):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    
    losses = []
    with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
        epoch_loss = 0
        for epoch in tepoch:
            for batch_index, training_sample in enumerate(train_loader):
                log_prob = model.log_probability(training_sample)
                loss = - log_prob.mean(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            epoch_loss /= len(train_loader)
            losses.append(np.copy(epoch_loss.detach().numpy()))
            tepoch.set_postfix(loss=epoch_loss.detach().numpy())

    return model, losses


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
            mean = kwargs.pop('mean', [0, 3])
            cov = kwargs.pop('mean', np.diag([.1, .1]))
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


if __name__ == '__main__':
    num_layers= 4
    masks = torch.nn.functional.one_hot(torch.tensor([i % 2 for i in range(num_layers)])).float()
    hidden_size = 32

    data = FlowDataset('MultipleGaussians', num_gaussians=5)
    NVP_model = RealNVP(masks, hidden_size)
    flow_model, loss = train(NVP_model, data, epochs = 100)

    plot_density(flow_model)