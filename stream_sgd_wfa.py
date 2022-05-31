import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from ignite.contrib.metrics import *
from hmmlearn import hmm
from Dataset import *
from torch import optim
from neural_density_estimation import hankel_density, ground_truth_hmm
import pickle
import sys
import gen_gmmhmm_data as ggh
import torch.distributions as D
from torch.distributions import Normal, mixture_same_family
import argparse
import os
import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt
from gradient_descent import *
from utils import phi, encoding,Fnorm, MAPE, phi_predict
from Density_WFA import density_wfa
from NN_learn_transition import NN_transition_WFA
from utils import *
# torch.autograd.set_detect_anomaly(True)

from nonstationary_HMM import incremental_HMM
def one_hot(y, num_classes = 2):
    tmp = torch.ones(num_classes)*(0)
    tmp[y] = 1
    return tmp

class stream_RNADE_RNN(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.input_size = parameters['input_size']
        self.RNN_hidden_size = parameters['RNN_hidden_size']
        self.RNN_num_layers = parameters['RNN_num_layers']
        self.device = parameters['device']
        self.num_classes = parameters['num_classes']
        self.mixture_number = parameters['mixture_number']
        self.xd = self.input_size
        self.initial_bias = None
        self.model = parameters['model']
        self.evaluate_interval =  parameters['evaluate_interval']
        self.smoothing_factor = parameters['smoothing_factor']


        print('here', self.input_size, self.RNN_hidden_size, self.RNN_num_layers)
        if self.model == 'lstm':
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.RNN_hidden_size, num_layers=self.RNN_num_layers, batch_first=True)
        else:
            self.lstm = nn.GRU(input_size=self.input_size, hidden_size=self.RNN_hidden_size,
                                num_layers=self.RNN_num_layers, batch_first=True)



        if self.num_classes is not None:
            self.mu_outs = nn.ModuleList()
            self.mu_outs2 = nn.ModuleList()
            self.sig_outs = nn.ModuleList()
            self.alpha_outs = nn.ModuleList()
            for i in range(self.num_classes):
                self.mu_outs.append(torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True))
                self.mu_outs2.append(torch.nn.Linear(self.mixture_number*self.input_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True))
                self.sig_outs.append(torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True))
                self.alpha_outs.append(torch.nn.Linear(self.RNN_hidden_size, self.mixture_number, bias=True).requires_grad_(True))
        else:
            self.mu_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True)
            self.sig_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True)
            self.alpha_out = torch.nn.Linear(self.RNN_hidden_size, self.mixture_number, bias=True).requires_grad_(True)
            self.mu_out2 = torch.nn.Linear(self.mixture_number*self.input_size, self.mixture_number*self.input_size, bias=True).requires_grad_(True)
        self.to(device)


    def forward(self, prev_state, x, prediction = False):
        x = x.to(device).float()
        x = x.reshape(1, 1, X.shape[-1])
        if self.model == 'lstm':
            state = prev_state[0]
            tmp_result = phi(self, x, state, prediction)
            output, (state_h, state_c) = self.lstm(x, prev_state)
            prev_state = (state_h.detach(), state_c.detach())
            return tmp_result, prev_state
        else:
            tmp_result = phi(self, x, prev_state, prediction)
            output, state_h = self.lstm(x, prev_state)
            prev_state = torch.sigmoid(state_h.detach())
            return tmp_result, prev_state

    def init_state(self, N):
        if args.method == 'lstm':
            return (torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float(),
                    torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float())
        else:
            return torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float()
    def lossfunc(self, prev, current, y = None):
        if y is None:
            log_likelihood = self(prev, current)
            log_likelihood = torch.mean(log_likelihood)
            # print(log_likelihood)
            return -log_likelihood
        else:
            class_weights, tmp = self(prev, current)
            loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
            y = y.reshape(-1)
            return loss(class_weights.reshape(1, -1), y), tmp

    def fit(self,train_x, optimizer, y = None, verbose = True, scheduler = None, pred_all = None, validation_number = 0):
        prev = self.init_state(1)
        joint_likelihood = 0.
        correct = 0
        total = 0
        results = []
        if pred_all is None:
            pred_all = []
        for i in range(validation_number, train_x.shape[0]):
            current_x = train_x[i].to(device)
            optimizer.zero_grad()

            pred_prob, _ = self(prev, current_x)
            loss, prev = self.lossfunc(prev, current_x, y[i])
            # if i<= 1000:
            #     loss = loss - 10 * one_hot(y[i], num_classes=2).to(self.device) @ pred_prob

            loss.backward()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)

            pred_class = torch.argmax(pred_prob)
            if pred_class == y[i]: correct += 1
            total += 1
            pred_all.append(torch.softmax(pred_prob, dim = 0).detach().cpu().numpy())

            if (i >= 300 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                # print(y[:(i+1)].shape)
                auc = sklearn.metrics.roc_auc_score(y[:(i+1)].cpu().numpy(), np.asarray(pred_all)[:, 1])
                print(i, loss.detach().cpu().numpy(), correct/total, pred_class, auc, pred_prob)
                pred_prob = pred_prob.detach().cpu().numpy()
                results.append([loss.detach().cpu().numpy(), correct / total, auc, pred_prob[0], pred_prob[1]])

        return results, pred_all

class stream_density_wfa(nn.Module):

    def __init__(self, xd, d, r, mix_n, device, task, evaluate_interval = 500, num_classes = None, use_batchnorm = True, init_std = 1e-3, double_pre = False, initial_bias = None, smoothing_factor = 0., seed= 1993):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.xd= xd
        self.d = d
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.encoder_1 = torch.nn.Linear(xd, d, bias=True)
        self.encoder_2 = torch.nn.Linear(d, d, bias=True)
        self.evaluate_interval = evaluate_interval
        self.r = r
        self.mix_n = mix_n
        self.smoothing_factor = smoothing_factor

        tmp_core = torch.normal(0, init_std, [r, d, r])
        self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        self.num_classes = num_classes
        # self.scale_likelihood = torch.nn.Linear(num_classes, num_classes, bias=True).requires_grad_(True)
        # self.scale = nn.Parameter(scale.clone().float().requires_grad_(True))
        if num_classes is not None and task == 'classification':
            self.mu_outs = nn.ModuleList()
            self.mu_outs2 = nn.ModuleList()
            self.sig_outs = nn.ModuleList()
            self.alpha_outs = nn.ModuleList()
            for i in range(num_classes):
                self.mu_outs.append(torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True))
                self.mu_outs2.append(torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True))
                self.sig_outs.append(torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True))
                self.alpha_outs.append(torch.nn.Linear(r, mix_n, bias=True).requires_grad_(True))
        else:
            self.num_classes = None
            self.mu_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
            self.sig_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
            self.alpha_out = torch.nn.Linear(r, mix_n, bias=True).requires_grad_(True)
            self.mu_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)

        self.batchnrom = nn.BatchNorm1d(self.A.shape[-1])
        self.double_pre = double_pre
        self.initial_bias = initial_bias
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)


    def forward(self, prev, current, next, prediction = False):
        if self.double_pre:
            prev = prev.double().to(device)
            current = current.double().to(device)
        else:
            prev = prev.float().to(device)
            current = current.float().to(device)
        current = self.encoder_1(current)
        current = torch.relu(current)
        current = self.encoder_2(current)
        current = torch.relu(current)
        tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
        tmp = torch.sigmoid(tmp)
        tmp_result = phi(self, next, tmp, prediction)

        return tmp_result, tmp.detach()

    def fit(self,train_x, optimizer, y = None, verbose = True, scheduler = None, pred_all = None, validation_number = 0, task = 'classification'):
        prev = torch.softmax(torch.rand([1, self.A.shape[0]]), dim = 1).to(device)
        joint_likelihood = 0.
        correct = 0
        total = 0
        results = []
        if pred_all is None:
            pred_all = []

        for i in range(validation_number, train_x.shape[0]-1):
            current_x = train_x[i].to(device)
            next = train_x[i+1].to(device)
            # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            optimizer.zero_grad()
            pred_prob, _ = self(prev, current_x, next)
            # if i >= 3000: reg_weight = 0.
            # else: reg_weight = 1.
            reg_weight = 1.
            if task =='regression':
                pred, _ =  self(prev, current_x, next, prediction = True)
                loss, prev = self.lossfunc(prev, current_x, next, y=train_x[i+1], reg_weight = reg_weight, task = task)
            else:
                loss, prev = self.lossfunc(prev, current_x, next, y=y[i+1], reg_weight=reg_weight, task=task)
            # print(self.A[0, 0, 0])
            loss.backward()
            prev = prev.detach()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            if task  == 'classification':
                pred_class = torch.argmax(pred_prob)
                if pred_class == y[i+1]: correct += 1
                total += 1
                pred_all.append(torch.softmax(pred_prob, dim = 0).detach().cpu().numpy())
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                    auc = sklearn.metrics.roc_auc_score(y[1:(i + 2)].cpu().numpy(), np.asarray(pred_all)[:, 1])
                    print(i, loss.detach().cpu().numpy(), correct / total, pred_class, auc, pred_prob)
                    pred_prob = pred_prob.detach().cpu().numpy()
                    results.append([loss.detach().cpu().numpy(), correct / total, auc, pred_prob[0], pred_prob[1]])
            elif task == 'regression':
                pred_all.append(pred.detach().cpu().numpy())
                current_mse = torch.mean((pred- train_x[i+1])**2).detach().cpu().numpy()
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                    print(i, np.mean(np.asarray(results)[:, -1]))

                results.append([loss.detach().cpu().numpy(), current_mse])

            # self.initial_bias = current_x
        return results, pred_all

    def eval_likelihood(self, X, batch = False):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = log_likelihood.detach().cpu().numpy()
        if not batch:
            return np.mean(log_likelihood)
        else:
            return log_likelihood

    def lossfunc(self, prev, current, next, task, y = None, reg_weight = 1.):
        if y is None:
            conditional_likelihood, tmp = self(prev, current, next)
            log_likelihood = torch.mean(conditional_likelihood)
            return -log_likelihood, tmp
        else:
            if task == 'classification':
                class_weights, tmp = self(prev, current, next)
                loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
                y = y.reshape(-1)
                one_hot_y = one_hot(y, num_classes=self.num_classes)
                reg = one_hot_y[0]*class_weights[0] + one_hot_y[1]*class_weights[1]
                return loss(class_weights.reshape(1, -1), y) - reg_weight*reg, tmp
                # return -reg_weight*reg, tmp
            elif task == 'regression':
                pred, tmp = self(prev, current, next, prediction = True)
                likelihood, tmp = self(prev, current, next, prediction = False)
                y = y.reshape(pred.shape)
                # return torch.mean((pred - y)**2), tmp
                return - likelihood , tmp


    def eval_prediction(self, X, y):
        if self.double_pre:
            y = y.double().to(device)
        else:
            y = y.float().to(device)
        pred_mu, pred_sig = self(X, prediction = True)
        pred_mu = pred_mu.reshape(y.shape)
        print(pred_mu[0], y[0])
        return MAPE(pred_mu, y), torch.mean(pred_sig), torch.mean((pred_mu - y)**2), torch.mean(y**2), torch.mean(pred_mu**2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default=64, type=int, help='hidden states size of the model')
    parser.add_argument('--exp_data', default='weather', help='dataset for the experiment')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--nc', default=2, type=int, help='number of classes')
    parser.add_argument('--mix_n', default=20, type=int, help='number of mixture components')
    parser.add_argument('--method', default='gru', help='method to use')
    parser.add_argument('--task', default='classification', help='task to perform')
    return parser

def normalize(X):
    tmp = X[:int(len(X))]
    mean = np.mean(tmp, axis = 0)
    std = np.std(tmp, axis = 0)
    return (X  - mean)/std

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    evaluate_interval = 1000
    validation_number = 200000
    if args.exp_data == 'weather':
        X, y = get_weather()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'weather')
    elif args.exp_data == 'ETT':
        X = get_ETT()
        X = normalize(X)
        y = np.ones(X.shape[0])
        file_dir = os.path.join(file_dir, 'realWorld', 'ETT-small')
    elif args.exp_data == 'sea':
        X, y = get_sea_data()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'sea')
    elif args.exp_data == 'elec':
        X, y = get_electronic_data()
        X = X[:, 2:]
        # X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'Elec2')
    elif args.exp_data == 'mixeddrift':
        X, y = get_mixeddrift()
        file_dir = os.path.join(file_dir, 'artificial', 'mixedDrift')
    elif args.exp_data == 'hyperplane':
        X, y = get_hyperplane()
        file_dir = os.path.join(file_dir, 'artificial', 'hyperplane')
    elif args.exp_data == 'chess':
        X, y = get_chess()
        file_dir = os.path.join(file_dir, 'artificial', 'chess')
    elif args.exp_data == 'covType':
        X, y = get_covtype()
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 1000
    elif args.exp_data == 'rialto':
        X, y = get_rialto()
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'poker':
        X, y = get_poker()
        file_dir = os.path.join(file_dir, 'realWorld', 'poker')
        evaluate_interval = 1000
    if args.task == 'classification':
        y = torch.tensor(y).type(torch.LongTensor).to(device)
        y = torch.tensor(y).reshape(-1, 1).to(device)
    X = torch.tensor(X).to(device)
    print(X.shape)
    # import time
    # time.sleep(2)
    #
    # nshmm = incremental_HMM(r = 3)
    # X = nshmm.sample(1000)
    # X = np.asarray(X).swapaxes(0, 1)
    # ground_truth_joint, ground_truth_conditionals = nshmm.score(X, stream = True)
    # ground_truth_conditionals = np.asarray(ground_truth_conditionals)
    # ground_truth_joint = np.asarray(ground_truth_joint)

    lr = args.lr
    validation_results = {}
    all_mix_ns = [20, 16, 32, 64, 128, 256, 512]
    smoothing_factors = [0]
    seeds = [1993]
    for r in [20, 16, 32, 64, 128, 256, 512]:
        for mix_n in all_mix_ns:
            for sf in smoothing_factors:
                for seed in seeds:
                    if args.method == 'lstm' or args.method == 'gru':
                        default_parameters = {
                            'input_size': X.shape[-1],
                            'RNN_hidden_size': r,
                            'RNN_num_layers': 1,
                            'device': device,
                            'num_classes': args.nc,
                            'mixture_number': mix_n,
                            'model': args.method,
                            'evaluate_interval': evaluate_interval,
                            'smoothing_factor': sf,
                            'task': args.task
                        }
                        #
                        model = stream_RNADE_RNN(default_parameters)
                    elif args.method == 'wfa':
                        model = stream_density_wfa(xd=X.shape[1], num_classes=args.nc, d=r, r=r, mix_n=mix_n, device=device,
                                                   initial_bias=None, evaluate_interval = evaluate_interval, smoothing_factor = sf, seed= seed, task=args.task)

                    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
                    results, pred_all = model.fit(X[:validation_number], optimizer, y=y[:validation_number], verbose=True, scheduler=None, task=args.task)

                    if r not in validation_results:
                        validation_results[r] = {}
                        validation_results[r]['model'] = [model]
                        validation_results[r]['pred_all'] = [pred_all]
                        validation_results[r]['final_auc'] = [results[-1][-1]]
                    else:
                        validation_results[r]['model'].append(model)
                        validation_results[r]['pred_all'].append(pred_all)
                        validation_results[r]['final_auc'].append(results[-1][-1])
    # print(validation_results)
    mix_ns = {}
    max_auc = 0.
    final_r = 1
    model = None
    pred_all = None
    for r in validation_results.keys():
        for i, auc in enumerate(validation_results[r]['final_auc']):
            if auc >= max_auc:
                max_auc = auc
                model = validation_results[r]['model'][i]
                pred_all = validation_results[r]['pred_all'][i]
    #     mix_ns[r] = (np.argmax(np.asarray(validation_results[r]['final_auc'])))
    #     tmp = np.max(np.asarray(validation_results[r]['final_auc']))
    #     # print(tmp, max_auc)
    #     if max_auc < tmp:
    #         max_auc = tmp
    #         model = validation_results[r]['model'][mix_ns[r]]
    #         pred_all = validation_results[r]['pred_all'][mix_ns[r]]
    #         final_r = r
    # # final_mix_n = all_mix_ns[mix_ns[final_r]]
    # print(final_r, final_mix_n)
    try:
        args.r = model.r
        args.mix_n = model.mix_n
        print('wfa', model.r, model.mix_n)

    except:
        args.r = model.RNN_hidden_size
        args.mix_n = model.mixture_number
        print('rnn', model.RNN_hidden_size, model.mixture_number)
    # args.r = model.r
    # args.mix_n = final_mix_n
    #
    # if args.method == 'lstm' or args.method == 'gru':
    #     default_parameters = {
    #         'input_size': X.shape[-1],
    #         'RNN_hidden_size': final_r,
    #         'RNN_num_layers': 1,
    #         'device': device,
    #         'num_classes': args.nc,
    #         'mixture_number': final_mix_n,
    #         'model': args.method,
    #         'evaluate_interval': evaluate_interval
    #     }
    #     #
    #     model = stream_RNADE_RNN(default_parameters)
    # elif args.method == 'wfa':
    #     model = stream_density_wfa(xd=X.shape[1], num_classes=args.nc, d=X.shape[1], r=final_r, mix_n=final_mix_n, device=device,
    #                                initial_bias=None, evaluate_interval=evaluate_interval)

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    results, pred_all = np.asarray(model.fit(X, optimizer, y = y, validation_number = validation_number, verbose=True, scheduler = None, pred_all = pred_all, task=args.task))
    print(results)
    save_file = {'args': args, 'results': results}
    with open(os.path.join((file_dir), f'{args.method}_results'), 'wb') as f:
        pickle.dump(save_file, f)

    with open(os.path.join((file_dir), f'{args.method}_prediction'), 'wb') as f:
        pickle.dump(pred_all, f)


