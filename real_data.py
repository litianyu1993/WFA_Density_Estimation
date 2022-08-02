import pickle
import argparse
from RNADE_transformer import run_transformer
import numpy as np
import torch
from Dataset import *
from SGD_WFA import learn_SGD_WFA
from gradient_descent import train, validate
from torch import optim
from RNADE_RNN import RNADE_RNN
from matplotlib import pyplot as plt
from Density_WFA_finetune import learn_density_WFA, density_wfa_finetune
from neural_density_estimation import hankel_density, ground_truth_hmm
from matplotlib import pyplot as plt
from hmmlearn import hmm
import os
from utils import exp_parser, sliding_window, autoregressive_regression
from nonstationary_HMM import incremental_HMM
from HMM_experiments import WFA_SGD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def standardize(X, mean = None, std = None):
    if mean is None:
        mean = np.mean(X, axis = 0)
    if std is None:
        std = np.std(X, axis = 0)
    return (X - mean)/std, mean, std

def difference(X):
    print(X.shape)
    min_X = np.min(X, axis = 0)

    X = (X - min_X)+1
    X = np.log(X)
    for i in range(len(X)-1):
        X[i] = X[i+1] - X[i]
    return X[:-1]

def recover(X, x):

    for i in range(1, len(X)):
        X[-i] = x - X[-i]
        x = X[-i]
    return X[1:]

def evaluate(model, method, exp_name, r, N, X, fileDir,load_test = False, save_test = False):
    ls = np.arange(1, 20)*8
    # print(ls)
    test_data = {}
    if not load_test:
        for l in ls:
            test_data[l] = sliding_window(X, l)
        # file_name = os.path.join(fileDir, exp_name+'test_data')
        print(fileDir)
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
        with open(fileDir + exp_name+'test_data', 'wb') as f:
            pickle.dump(test_data, f)
    else:
        file_name = os.path.join(fileDir, exp_name+'test_data')
        if not os.path.exists(fileDir):
            os.makedirs(fileDir )
        with open(file_name, 'rb') as f:
            test_data = pickle.load(f)

    results = {}
    results['exp_parameters'] = args
    for l in ls:
        train_x = torch.tensor(test_data[l]).float()
        if method == 'WFA' or method == 'SGD_WFA':
            likelihood = model.eval_likelihood(train_x)
        elif method == 'LSTM':
            likelihood = model.eval_likelihood(torch.swapaxes(train_x, 1, 2))
        elif method == 'Transformer':
            data = torch.swapaxes(train_x, 1, 2)
            generator_params = {'batch_size': 256,
                                'shuffle': True,
                                'num_workers': 0}
            test_loader = torch.utils.data.DataLoader(data, **generator_params)

            counts = 0
            likelihood = 0
            for x in test_loader:
                x = torch.transpose(x, 0, 1)
                x = x.to(device)
                test_loss = model.eval_likelihood(x)
                likelihood += test_loss.detach().cpu().numpy()
                counts += 1
            likelihood /= counts
        results[l] = {
            'model_output': likelihood
        }
        print("Length" + str(l) + "result is:")
        print("Model output: " + str(likelihood))
        # file_name = os.path.join(fileDir, exp_name+'results')
        # if not os.path.exists(fileDir):
        #     os.makedirs(fileDir)
        # print(file_name)
        with open(os.path.join(fileDir, exp_name + 'results'), 'wb') as f:
            pickle.dump(results, f)
    return



if __name__ == '__main__':
    parser = exp_parser()
    args = parser.parse_args()
    noise = args.noise

    nn_trainsition = args.nn_transition

    load_test = args.load_data
    np.random.seed(args.seed)
    batch_size = args.batch_size
    r = args.r
    mix_n = args.mix_n
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
    regression_lr = args.regression_lr
    index_run = args.run_idx
    regression_epochs = args.regression_epochs
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    data_folder =  os.path.join(fileDir, args.exp_data)
    exp_folder = os.path.join(data_folder, 'rank_'+str(r))
    # file_name = os.path.join(fileDir, args.exp_data ,'rank_'+str(r))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    exp_name = method + 'run_idx'+str (args.seed) +'N' +str(args.N)+'noise_'+str(args.noise)+'L_'+str(args.L)
    print(exp_name)
    if args.exp_data == 'weather':
        X = np.genfromtxt(os.path.join(fileDir, args.exp_data, 'NEweather_data.csv'), delimiter=',')

        if N > X.shape[0] - 1000:
            N = X.shape[0]-1000
            args.N = N
        X = X[:args.N]
        print(X.shape)
        test_X = np.genfromtxt(os.path.join(fileDir, args.exp_data, 'NEweather_data.csv'), delimiter=',')[args.N:args.N + 1000]
    elif args.exp_data == 'movingSquares':
        data = np.genfromtxt(os.path.join(fileDir, args.exp_data, 'movingSquares.data'), delimiter=' ')
        X = data[170000: 170000 + args.N]
        print(X.shape)
        test_X = data[-1000:]
    elif args.exp_data == 'Elec2':
        data = np.genfromtxt(os.path.join(fileDir, args.exp_data, 'elec2_data.dat'),
                             skip_header=1,
                             skip_footer=1,
                             names=True,
                             dtype=float,
                             delimiter=' ')
        new_data = []
        for i in range(len(data)):
            x = []
            for j in range(len(data[i])):
                x.append(data[i][j])
            new_data.append(x)
        new_data = np.asarray(new_data)
        new_data = difference(new_data)
        if N > new_data.shape[0] - 1000:
            N = new_data.shape[0]-1000
            args.N = N
        X = new_data[:args.N]
        print(X.shape)
        test_X = new_data[args.N:args.N + 1000]

    # X = difference(X)
    # test_X = difference(test_X)
    auto_MAPE, auto_MSE = autoregressive_regression(X, test_X)
    bias = torch.tensor(np.mean(X, axis = 0))
    xd = X.shape[-1]
    if method == 'WFA' or method == 'SGD_WFA':
        model_params = {
            'd': xd,
            'xd': xd,
            'r': r,
            'mixture_n': mix_n,
            'lr': hankel_lr,
            'l': args.L,
            'epochs': hankel_epochs,
            'batch_size': 256,
            'fine_tune_epochs': finetuen_epochs,
            'fine_tune_lr': finetune_lr,
            'double_precision': False,
            'verbose': True,
            'nn_transition': nn_trainsition,
            'GD_linear_transition': False,
            'use_softmax_norm': not args.wfa_sgd_batchnorm,
            'regression_lr': regression_lr,
            'regression_epochs': regression_epochs,
            'use_batch_norm': args.wfa_sgd_batchnorm
        }
        #
        data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
        DATA = {}
        Ls = [l, 2*l, 2*l+1]
        for k in range(len(Ls)):
            L = Ls[k]
            train_x = sliding_window(X, L)
            test_x = sliding_window(test_X, L)
            train_x = torch.tensor(train_x).float()
            test_x = torch.tensor(test_x).float()
            print(train_x.shape, test_x.shape)
            DATA[data_label[k][0]] = train_x
            DATA[data_label[k][1]] = test_x
        if method == 'WFA':
            dwfa_finetune = learn_density_WFA(DATA, model_params, l, plot=False)
        else:
            model_params['epochs'] = 0
            dwfa_finetune = learn_SGD_WFA(model_params, DATA, initial_bias=bias)

        model_name = os.path.join(exp_folder, exp_name + 'model')


        torch.save(dwfa_finetune.state_dict(), model_name)
        evaluate(model=dwfa_finetune, X= test_X, method=method, exp_name=exp_name, r=r, fileDir=exp_folder, load_test=load_test, N = args.N)

        test_x = torch.tensor(sliding_window(test_X, 500)).to(device)
        # print(test_x.shape)
        print(dwfa_finetune.eval_prediction(test_x[:, :, :-1],test_x[:, :, -1]))
        wfa_mape, wfa_mse = dwfa_finetune.bootstrapping(test_x)
        plt.plot(wfa_mape, label = 'wfa')
        plt.plot(auto_MAPE, label = 'auto')
        plt.legend()
        plt.show()

        plt.plot(wfa_mse, label = 'wfa')
        plt.plot(auto_MSE, label = 'auto')
        plt.legend()
        plt.show()


    elif method == 'Transformer':
        l = 2*l+1
        train_x = sliding_window(X, l).swapaxes(1, 2)
        print(train_x.shape)
        test_x = sliding_window(test_X, l).swapaxes(1, 2)
        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()
        default_parameters = {
            'input_size': X.shape[-1],
            'lr': args.transformer_lr,
            'epochs': args.transformer_epochs,
            'batch_size': args.batch_size,
            'mixture_n': args.hmm_rank,
            'device': device,
            'nhead': args.nhead,
            'initial_bias': bias
        }
        tran_model = run_transformer(train_x, test_x, default_parameters, verbose=True, seed = args.seed).to(device)
        evaluate(model=tran_model, X=test_X, method=method, exp_name=exp_name, r=r, fileDir=exp_folder, load_test=load_test,
                 N=args.N)
        file_name_tmp = os.path.join(exp_folder, exp_name + 'model')
        torch.save(tran_model.state_dict(), file_name_tmp)



    elif method == 'flow':
        from flows import TorchDataset, BNAF
        from flows import train as train_flows

        results = {}
        results['exp_parameters'] = args

        for l in np.arange(1, 20)*8:
        # for l in [152]:
            train_x = sliding_window(X, l)
            test_x = sliding_window(test_X, l)
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)


            test_x = train_x
            test_x = torch.tensor(test_x).float()

            n_input = train_x.shape[1]
            n_layers = 2
            n_hidden = 2*n_input
            train_data = TorchDataset(train_x)
            test_data = TorchDataset(test_x)
            # train_data = RealNVP(n_input=n_input, n_layers=n_layers, n_hidden=n_hidden)
            bnaf_model = BNAF(n_input=n_input, n_layers=n_layers, n_hidden=n_hidden, seed = args.seed)
            flow_model, loss, test_loss = train_flows(bnaf_model, {'train': train_data, 'test': test_data}, epochs=200)

            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)
            # loss = np.mean(np.asarray(loss))
            results[l] = {'model_output': loss[-1], 'loss': test_loss}
            print(results)
            with open(os.path.join(exp_folder, exp_name + 'results'), 'wb') as f:
                pickle.dump(results, f)

            file_name_tmp = os.path.join(exp_folder, exp_name + 'model'+'L_'+str(l))
            torch.save(flow_model.state_dict(), file_name_tmp)


    elif method == 'LSTM':
        l = 2*l+1
        default_parameters = {
            'input_size': X.shape[-1],
            'RNN_hidden_size': r,
            'RNN_num_layers': 1,
            'output_size': r,
            'mixture_number': r,
            'device': device,
            'initial_bias': bias
        }
        train_x = sliding_window(X, l).swapaxes(1, 2)
        test_x = sliding_window(test_X, l).swapaxes(1, 2)
        train_x = torch.tensor(train_x).float()
        test_x = torch.tensor(test_x).float()

        train_data = Dataset(data=[train_x])
        test_data = Dataset(data=[test_x])

        generator_params = {'batch_size': batch_size,
                            'shuffle': False,
                            'num_workers': 0}
        train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_data, **generator_params)
        rnade_rnn = RNADE_RNN(default_parameters)
        optimizer = optim.Adam(rnade_rnn.parameters(), lr=lstm_lr, amsgrad=True)
        # likelihood = rnade_rnn(train_x)
        # print(torch.mean(torch.log(likelihood)))
        train_likeli, test_likeli = rnade_rnn.fit(train_x, test_x, train_loader, test_loader, lstm_epochs, optimizer,
                                                  scheduler=None,
                                                  verbose=True)
        evaluate(model=rnade_rnn, X=test_X, method=method, exp_name=exp_name, r=r, fileDir=exp_folder, load_test=load_test, N = args.N)
        file_name_tmp = os.path.join(exp_folder, exp_name + 'model')
        torch.save(rnade_rnn.state_dict(), file_name_tmp)