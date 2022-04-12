import pickle
import argparse
import numpy as np
import torch
from Dataset import *
from gradient_descent import train, validate
from torch import optim
from RNADE_RNN import RNADE_RNN
from Density_WFA_finetune import learn_density_WFA, density_wfa_finetune
from neural_density_estimation import hankel_density, ground_truth_hmm, insert_bias
from matplotlib import pyplot as plt
from hmmlearn import hmm
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def WFA_SGD(data, model_params, l, plot=True, out_file_name=None, load_WFA=False, load_hankel=False,
                          singular_clip_interval=10, file_path=None):
        # data comes in a dictionary, with keys: train_l, train_2l, train_2l1, test_l, test_2l, test_2l1 indicating the length of the data
        # all data are in torch.tensor form
        # model_params is also a dictionary with keys: d (encoder output dimension), xd (original dimension of the input data), r, mixture_n, double_precision
        Ls = [l, 2 * l, 2 * l + 1]
        data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
        hds = []
        lr, epochs, batch_size, double_precision = model_params['lr'], model_params['epochs'], model_params[
            'batch_size'], model_params['double_precision']
        d, xd, r, mixture_n = model_params['d'], model_params['xd'], model_params['r'], model_params['mixture_n']
        nn_transition = model_params['nn_transition']
        GD_linear_transition = model_params['GD_linear_transition']
        verbose = model_params['verbose']
        generator_params = {'batch_size': batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        hd = hankel_density(d, xd, r, mixture_number=mixture_n, L=l, double_pre=double_precision,
                            nn_transition=nn_transition, GD_linear_transition=GD_linear_transition).cuda(device)
        dwfa_finetune = density_wfa_finetune(hankel=hd, double_pre=double_precision, nn_transition=nn_transition,
                                             GD_linear_transition=GD_linear_transition)
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
            train_loader = torch.utils.data.DataLoader(train_x, **generator_params)
            test_loader = torch.utils.data.DataLoader(test_x, **generator_params)

            optimizer = optim.Adam(dwfa_finetune.parameters(), lr=fine_tune_lr, amsgrad=True)
            train_likeli, test_likeli = dwfa_finetune.fit(train_x, test_x, train_loader, test_loader, fine_tune_epochs,
                                                          optimizer, scheduler=None,
                                                          verbose=verbose,
                                                          singular_clip_interval=singular_clip_interval)
            if plot:
                plt.plot(train_likeli, label='train')
                plt.plot(test_likeli, label='test')
                plt.legend()
                plt.show()

        return dwfa_finetune

def evaluate(model, hmmmodel, method, exp_name, r, N, fileDir,load_test = False, xd = 1):
    ls = np.arange(1, 50)*8
    print(ls)
    test_data = {}
    if not load_test:
        for l in ls:
            train_x = np.zeros([1000, xd, l])
            for i in range(1000):
                x, z = hmmmodel.sample(l)
                train_x[i, :, :] = x.reshape(xd, -1)
            test_data[l] = train_x
        file_name = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'test_data_N'+str(N))
        with open(file_name, 'wb') as f:
            pickle.dump(test_data, f)
    else:
        file_name = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'test_data_N'+str(N))
        with open(file_name, 'rb') as f:
            test_data = pickle.load(f)

    results = {}
    results['exp_parameters'] = args
    for l in ls:
        train_ground_truth = ground_truth_hmm(test_data[l], hmmmodel)
        train_x = torch.tensor(test_data[l]).float()
        if method == 'WFA' or method == 'SGD_WFA':
            likelihood = model.eval_likelihood(train_x)
        else:
            likelihood = rnade_rnn.eval_likelihood(torch.swapaxes(train_x, 1, 2))
        results[l] = {
            'model_output': likelihood,
            'ground_truth': train_ground_truth
        }
        print("Length" + str(l) + "result is:")
        print("Model output: " + str(likelihood) + "Ground truth: " + str(train_ground_truth))
        file_name = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', exp_name)
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)
    return


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
    parser.add_argument('--seed', default = 1993, type = int, help='random seed')
    parser.add_argument('--load_test_data', dest='load_data', action='store_true')
    parser.add_argument('--new_test_data', dest='load_data', action='store_false')
    parser.add_argument('--noise', default=0, type = float, help='variance of the added noise')
    parser.add_argument('--nn_transition', dest='nn_transition', action='store_true')
    parser.add_argument('--no_nn_transition', dest='nn_transition', action='store_false')
    parser.set_defaults(load_data=True)
    args = parser.parse_args()
    noise = args.noise

    nn_trainsition = args.nn_transition

    load_test = args.load_data
    np.random.seed(args.seed)
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
    exp_name = method + 'run_idx'+str (args.seed) +'N' +str(args.N)+'noise_'+str(args.noise)
    if not os.path.exists(file_name + 'exp'):
        os.makedirs(file_name + 'exp')

    with open(file_name, 'rb') as f:
        hmmmodel = pickle.load(f)

    if method == 'WFA' or method == 'SGD_WFA':
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
            'nn_transition': nn_trainsition,
            'GD_linear_transition': False,
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
                train_x[i, :, :] = x.reshape(xd, -1) + np.random.normal(0, noise, [xd, L])
                x, z = hmmmodel.sample(L)
                test_x[i, :, :] = x.reshape(xd, -1)+ np.random.normal(0, noise, [xd, L])

            train_x = torch.tensor(train_x).float()
            test_x = torch.tensor(test_x).float()
            DATA[data_label[k][0]] = train_x
            DATA[data_label[k][1]] = test_x
        if method == 'WFA':
            dwfa_finetune = learn_density_WFA(DATA, model_params, l, plot=False)
        else:
            model_params['epochs'] = 0
            dwfa_finetune = WFA_SGD(DATA, model_params, l, plot=False)
        evaluate(model=dwfa_finetune, hmmmodel=hmmmodel, method=method, exp_name=exp_name, r=r, fileDir=fileDir, load_test=load_test, N = args.N)

    elif method == 'HMM':
        print(method)
        Ls = [l, 2 * l, 2 * l + 1]
        data = []
        lengths = []
        for k in range(len(Ls)):
            L = Ls[k]
            train_x = np.zeros([N, L, xd])
            for i in range(N):
                x, z = hmmmodel.sample(L)
                train_x[i, :, :] = x.reshape(L, -1) + np.random.normal(0, noise, [L, xd])
                lengths.append(L)
            data.append(train_x)

        remodel = hmm.GaussianHMM(n_components=r, covariance_type="full", n_iter=200)
        train_x = np.concatenate((data[0].reshape(N*l, xd),data[1].reshape(N*l*2, xd),  data[2].reshape(N*(l*2+1), xd)), axis = 0)
        remodel.fit(train_x, lengths)
        file_name = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'test_data_N' + str(N))
        with open(file_name, 'rb') as f:
            test_data = pickle.load(f)
        results = {}
        results['exp_parameters'] = args
        ls = np.arange(1, 50) * 8
        for l in ls:
            train_ground_truth = ground_truth_hmm(test_data[l], hmmmodel)
            x = test_data[l]
            x = np.swapaxes(x, 1,2)
            testx = x.reshape(x.shape[0]*x.shape[1], -1)
            likelihood = remodel.score(testx, lengths = np.ones(x.shape[0]).astype(int)*l)/x.shape[0]
            results[l] = {
                'model_output': likelihood,
                'ground_truth': train_ground_truth
            }
            print("Length" + str(l) + "result is:")
            print("Model output: " + str(likelihood) + "Ground truth: " + str(train_ground_truth))
            file_name = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', exp_name)
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)




    elif method == 'LSTM':
        N = 3*N
        l = 2*l+1
        default_parameters = {
            'input_size': 1,
            'RNN_hidden_size': r,
            'RNN_num_layers': 1,
            'output_size': r,
            'mixture_number': r,
            'device': device
        }
        train_x = np.zeros([N, l, xd])
        test_x = np.zeros([N, l, xd])

        for i in range(N):
            x, z = hmmmodel.sample(l)
            train_x[i, :, :] = x.reshape(l, xd)+ np.random.normal(0, noise, [l, xd])
            x, z = hmmmodel.sample(l)
            test_x[i, :, :] = x.reshape(l, xd)+ np.random.normal(0, noise, [l, xd])
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
        evaluate(model=rnade_rnn, hmmmodel=hmmmodel, method=method, exp_name=exp_name, r=r, fileDir=fileDir, load_test=load_test, N = args.N)



