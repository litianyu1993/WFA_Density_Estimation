import torch
import torch.nn as nn
import numpy as np
import time
import math
import gen_gmmhmm_data as ggh
from gradient_descent import train, validate
from neural_density_estimation import ground_truth_hmm
from matplotlib import pyplot
from utils import torch_mixture_gaussian
torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, input_size = 10, feature_size=10, num_layers=1, dropout=0.1, mixture_number = 10, nhead = 1, seed = 1993, initial_bias = None):
        super(TransAm, self).__init__()
        torch.manual_seed(seed)
        self.input_size  =input_size
        self.model_type = 'Transformer'
        self.encoder1 = torch.nn.Linear(input_size, feature_size, bias=True)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.initial_bias = initial_bias


        self.mu_out = torch.nn.Linear(feature_size, mixture_number * input_size, bias=True)
        self.sig_out = torch.nn.Linear(feature_size, mixture_number * input_size, bias=True)
        self.alpha_out = torch.nn.Linear(feature_size, mixture_number, bias=True)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.mu_out.bias.data.zero_()
        self.mu_out.weight.data.uniform_(-initrange, initrange)
        self.sig_out.bias.data.zero_()
        self.sig_out.weight.data.uniform_(-initrange, initrange)
        self.alpha_out.bias.data.zero_()
        self.alpha_out.weight.data.uniform_(-initrange, initrange)


    def forward(self, src):
        encoded_src = self.encoder1(src)
        encoded_src = torch.relu(encoded_src)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask


        pos_src = self.pos_encoder(encoded_src)
        output = self.transformer_encoder(pos_src, self.src_mask)  # , self.src_mask)
        mu = self.mu_out(output)
        mu = mu.reshape(mu.shape[0], mu.shape[1], -1, self.input_size)
        if self.initial_bias is not None:
            for i in range(mu.shape[-1]):
                mu[:, :, :, i] = mu[:, :, :, i] + self.initial_bias[i]

        sig = torch.exp(self.sig_out(output))
        alpha = torch.softmax(self.alpha_out(output), dim = 2)
        log_like = 0.
        for i in range(mu.shape[0]):
            tmp = torch_mixture_gaussian(src[i, :, :], mu[i, :, :, :].reshape(src.shape[1], -1, src.shape[-1]), sig[i, :, :].reshape(src.shape[1], -1, src.shape[-1]), alpha[i, :, :])
            log_like += tmp
        return log_like

    def lossfunc(self, src):
        return -torch.mean(self(src))

    def eval_likelihood(self, src):
        return torch.mean(self(src.to(device)))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None, data_path = None):
        'Initialization'
        if data is not None:
            self.x = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[1]

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.x[:, index, :]
        return x


def run_transformer(train_data, test_data, model_params, verbose = True, seed = 1993):
    lr, epochs, batch_size = model_params['lr'], model_params['epochs'], model_params['batch_size']
    input_size, mixture_n, nhead, initial_bias = model_params['input_size'], model_params['mixture_n'],  model_params['nhead'], model_params['initial_bias']
    generator_params = {'batch_size': 256,
                        'shuffle': True,
                        'num_workers': 0}
    print(train_data.shape)
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)
    model = TransAm(input_size= input_size, feature_size=2*input_size, mixture_number=mixture_n, nhead = nhead, seed = seed, initial_bias=initial_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float("inf")
    # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):

        epoch_start_time = time.time()
        loss = train(model, device, train_loader, optimizer, train_data, transformer=True)
        vali_loss = validate(model, device, test_loader, optimizer, test_data, transformer=True)
        if vali_loss <= best_val_loss:
            best_val_loss = vali_loss
            best_model = model
        scheduler.step(vali_loss)
        if verbose:
            print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(
                loss,
                vali_loss))
    return best_model



if __name__ == '__main__':
    feature_size = 1
    N = 1000
    l = 5
    mix_number = 3
    epochs = 1
    lr = 0.1
    _, _, hmmmodel = ggh.gen_gmmhmm_data(N=1, xd=feature_size, L=l, r=mix_number)

    train_x = np.zeros([l, N, feature_size])
    test_x = np.zeros([l, N,  feature_size])
    for i in range(N):
        x, z = hmmmodel.sample(l)
        train_x[:, i, :] = x.reshape(-1, feature_size)
        x, z = hmmmodel.sample(l)
        test_x[:, i, :] = x.reshape(-1, feature_size)

    train_x_torch = torch.tensor(train_x).float()
    test_x_torch = torch.tensor(test_x).float()
    train_data = TransformerDataset(train_x_torch)
    test_data = TransformerDataset(test_x_torch)
    generator_params = {'batch_size': 256,
                        'shuffle': True,
                        'num_workers': 0}

    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)
    model = TransAm(feature_size = feature_size, mixture_number= mix_number).to(device)


    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float("inf")
      # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):

        epoch_start_time = time.time()
        loss = train(model, device, train_loader, optimizer, train_data, transformer= True)
        vali_loss = validate(model, device, test_loader, optimizer, test_data, transformer=True)
        if vali_loss <= best_val_loss:
            best_val_loss = vali_loss
            best_model = model
        scheduler.step(vali_loss)
        if True:
            print('Epoch: ' + str(epoch) + 'Train Likelihood: {:.10f} Validate Likelihood: {:.10f}'.format(
                loss,
                vali_loss))
    hmm_data = np.swapaxes(test_x, 0, 1)
    print(hmm_data.shape)
    print(ground_truth_hmm(hmm_data, hmmmodel))
    # src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
    # out = model(src)
    #
    # print(out)
    # print(out.shape)