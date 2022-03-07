import pickle
import argparse
import numpy as np
import torch
from Dataset import *
from gradient_descent import train, validate
from torch import optim
from RNADE_RNN import RNADE_RNN
from Density_WFA_finetune import learn_density_WFA
from neural_density_estimation import hankel_density, ground_truth_hmm, insert_bias
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hmm_rank', default=2, type=int, help='Rank of the HMM')
    parser.add_argument('--method', default='WFA', help='method to use, either LSTM or WFA')
    parser.add_argument('--L', default=3, type=int,help='length of the trajectories, WFA takes L, 2L, 2L+1, LSTM takes 2L+1')
    parser.add_argument('--N', default= 100, type=int,help= 'number of examples to consider, WFA takes N, LSTM takes 3N')
    parser.add_argument('--xd', default=1, type=int,help= 'dimension of the input feature')
    parser.add_argument('--hankel_lr', default=0.01, type=float, help='hankel estimation learning rate')
    parser.add_argument('--fine_tune_lr', default=0.001, type = float, help='WFA finetune learning rate')
    parser.add_argument('--LSTM_lr', default=0.001, type=float, help='LSTM learning rate')
    parser.add_argument('--hankel_epochs', default=100, type=int, help='hankel estimation epochs')
    parser.add_argument('--fine_tune_epochs', default=100, type=int, help='WFA finetune epochs')
    parser.add_argument('--LSTM_epochs', default=100, type=int, help='WFA finetune epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--run_idx', default=0, type = int, help='index of the run')
    args = parser.parse_args()
    batch_size = args.batch_size
    r = args.hmm_rank
    method = args.method
    l = args.L
    N = args.N
    xd = args.xd
    hankel_lr = args.hankel_lr
    finetune_lr = args.fine_tune_lr
    lstm_lr = args.LSTM_lr
    lstm_epochs = args.LSTM_epochs
    hankel_epochs = args.hankel_epochs
    finetuen_epochs = args.fine_tune_epochs
    index_run = args.run_idx
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    file_name = os.path.join(fileDir, 'hmm_models' ,'rank_'+str(r))
    exp_name = method + 'run_idx'+str (index_run) +'N' +str(args.N)
    if not os.path.exists(file_name + 'exp'):
        os.makedirs(file_name + 'exp')

    with open(file_name, 'rb') as f:
        hmmmodel = pickle.load(f)

    if method == 'WFA':
        model_params = {
            'd': 1,
            'xd': xd,
            'r': r,
            'mixture_n': r,
            'lr': hankel_lr,
            'epochs': hankel_epochs,
            'batch_size': 256,
            'fine_tune_epochs': finetuen_epochs,
            'fine_tune_lr': finetune_lr,
            'double_precision': False,
            'verbose': True,
            'nn_transition': False,
            'GD_linear_transition': False
        }
        data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
        DATA = {}
        Ls = [l, 2*l, 2*l+1]
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

        ls = np.arange(2, 100)
        results = {}
        results['exp_parameters'] = args
        for l in ls:
            train_x = np.zeros([N, xd, l])
            # print(2*l)
            for i in range(N):
                x, z = hmmmodel.sample(l)
                train_x[i, :, :] = x.reshape(xd, -1)
            train_ground_truth = ground_truth_hmm(train_x, hmmmodel)
            train_x = torch.tensor(train_x).float()

            likelihood = dwfa_finetune.eval_likelihood(train_x)
            results[l] = {
                'model_output': torch.mean(likelihood),
                'ground_truth': train_ground_truth
            }
            print("Length" + str(l) + "result is:")
            print("Model output: " + str(torch.mean(likelihood)) + "Ground truth: " + str(train_ground_truth))
            file_name = os.path.join(fileDir, 'hmm_models','rank_'+str(r)+'exp',exp_name)
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)

    elif method == 'LSTM':
        N = 3*N
        l = 2*l+1
        default_parameters = {
            'input_size': 1,
            'RNN_hidden_size': r**2,
            'RNN_num_layers': 1,
            'output_size': r,
            'mixture_number': r,
            'device': device
        }
        train_x = np.zeros([N, l, xd])
        test_x = np.zeros([N, l, xd])

        for i in range(N):
            x, z = hmmmodel.sample(l)

            train_x[i, :, :] = x.reshape(l, xd)
            x, z = hmmmodel.sample(l)
            test_x[i, :, :] = x.reshape(l, xd)
        train_data = Dataset(data=[train_x])
        test_data = Dataset(data=[test_x])

        generator_params = {'batch_size': batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_data, **generator_params)
        rnade_rnn = RNADE_RNN(default_parameters)
        optimizer = optim.Adam(rnade_rnn.parameters(), lr=lstm_lr, amsgrad=True)
        # likelihood = rnade_rnn(train_x)
        # print(torch.mean(torch.log(likelihood)))
        train_likeli, test_likeli = rnade_rnn.fit(train_x, test_x, train_loader, test_loader, lstm_epochs, optimizer,
                                                  scheduler=None,
                                                  verbose=True)
        ls = np.arange(2, 100)
        results = {}
        results['exp_parameters'] = args
        for l in ls:
            train_x = np.zeros([N, l, xd])
            # print(2*l)
            for i in range(N):
                x, z = hmmmodel.sample(l)
                train_x[i, :, :] = x.reshape(-1, xd)
            train_ground_truth = ground_truth_hmm(train_x.swapaxes(1, 2), hmmmodel)
            train_x = torch.tensor(train_x).float()

            likelihood = -rnade_rnn.lossfunc(train_x)
            print("Length" + str(l) + "result is:")
            print("Model output: " + str(torch.mean(likelihood)) + "Ground truth: " + str(train_ground_truth))
            file_name = os.path.join(fileDir, 'hmm_models','rank_' + str(r) + 'exp', exp_name)
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)


