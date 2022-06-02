import numpy as np
import torch
from torch import nn
import sklearn
from utils import *
from gradient_descent import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from collections import Counter
from nonstationary_HMM import incremental_HMM
def one_hot(y, num_classes = 2):
    tmp = torch.ones(num_classes)*(0)
    tmp[y] = 1
    return tmp


class stream_density_wfa(nn.Module):

    def __init__(self, parameter_dict):
        super().__init__()
        torch.manual_seed(parameter_dict['seed'])
        self.task = parameter_dict['task']
        self.xd= parameter_dict['input_size']
        self.d = parameter_dict['encoding_size']
        self.device = parameter_dict['device']
        self.encoder_1 = torch.nn.Linear(self.xd, self.d, bias=True)
        self.encoder_2 = torch.nn.Linear(self.d, self.d, bias=True)
        self.evaluate_interval = parameter_dict['evaluate_interval']
        self.r = parameter_dict['hidden_size']
        self.mix_n = parameter_dict['mix_n']
        self.smoothing_factor = parameter_dict['smoothing_factor']
        self.init_std = parameter_dict['init_std']
        self.num_layers = parameter_dict['num_layers']


        self.num_classes = parameter_dict['num_classes']
        self.prior = torch.ones(self.num_classes)
        self.prior.to(self.device)
        self.model = parameter_dict['model']

        if self.model =='wfa':
            tmp_core = torch.normal(0, self.init_std, [self.r, self.d, self.r])
            self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        elif self.model == 'lstm':
            self.recurrent = nn.LSTM(input_size=self.d, hidden_size=self.r,
                                num_layers=self.num_layers, batch_first=True)
        elif self.model == 'gru':
            self.recurrent = nn.GRU(input_size=self.d, hidden_size=self.r,
                               num_layers=self.num_layers, batch_first=True)

        if self.num_classes is not None and self.task == 'classification':
            self.mu_outs = nn.ModuleList()
            self.mu_outs2 = nn.ModuleList()
            self.sig_outs = nn.ModuleList()
            self.alpha_outs = nn.ModuleList()
            for i in range(self.num_classes):
                self.mu_outs.append(torch.nn.Linear(self.r, self.mix_n * self.xd, bias=True).requires_grad_(True))
                self.mu_outs2.append(torch.nn.Linear(self.mix_n * self.xd,  self.mix_n *  self.xd, bias=True).requires_grad_(True))
                self.sig_outs.append(torch.nn.Linear( self.r,  self.mix_n *  self.xd, bias=True).requires_grad_(True))
                self.alpha_outs.append(torch.nn.Linear( self.r,  self.mix_n, bias=True).requires_grad_(True))
        else:
            self.num_classes = None
            self.mu_out = torch.nn.Linear( self.r,  self.mix_n *  self.xd, bias=True).requires_grad_(True)
            self.sig_out = torch.nn.Linear( self.r,  self.mix_n *  self.xd, bias=True).requires_grad_(True)
            self.alpha_out = torch.nn.Linear( self.r,  self.mix_n, bias=True).requires_grad_(True)
            self.mu_out2 = torch.nn.Linear( self.mix_n *  self.xd,  self.mix_n * self.xd, bias=True).requires_grad_(True)

        self.double_pre = parameter_dict['double_pre']
        self.initial_bias =  parameter_dict['initial_bias']
        self.window_size = parameter_dict['window_size']
        if self.double_pre:
            self.double().to(self.device)
        else:
            self.float().to(self.device)

    def get_prior(self, y, index, lag = 100):
        counts = torch.bincount(y[index - lag: index].reshape(-1))
        return counts/torch.sum(counts)
    def init_state(self, N):
        if self.model == 'lstm':
            return (torch.zeros(self.num_layers, N, self.r).to(device).float(),
                    torch.zeros(self.num_layers, N, self.r).to(device).float())
        else:
            return torch.zeros(self.num_layers, N, self.r).to(device).float()
    def forward(self, prev, X, prediction = False):
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)

        all_results = []
        for i in range(self.window_size - 1):
            assert self.window_size >= 2
            current = X[i]
            next = X[i+1]
            current = self.encoder_1(current)
            current = torch.relu(current)
            current = self.encoder_2(current)
            current = torch.relu(current)
            if self.model == 'wfa':
                tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
                tmp = torch.sigmoid(tmp)
                tmp_result = phi(self, next, tmp, prediction)
            elif self.model == 'lstm':
                output, (state_h, state_c) = self.recurrent(current.reshape([1, 1, -1]), prev)
                tmp = (state_h, state_c)
                tmp_result = phi(self, next, tmp[0].reshape(1, -1), prediction)
            elif self.model == 'gru':
                output, state_h = self.recurrent(current.reshape([1, 1, -1]), prev)
                tmp = torch.sigmoid(state_h.detach())
                tmp_result = phi(self, next, tmp.reshape(1, -1), prediction)
            all_results.append(tmp_result)
        if self.model == 'lstm':
            return tmp_result, (tmp[0].detach(), tmp[1].detach())
        else:
            return tmp_result, tmp.detach()

    def fit(self,train_x, optimizer, y = None, verbose = True, scheduler = None, pred_all = None, validation_number = 0, task = 'classification'):
        if self.model == 'wfa':
            prev = torch.softmax(torch.rand([1, self.A.shape[0]]), dim = 1).to(device)
        else:
            prev = self.init_state(1)
        joint_likelihood = 0.
        correct = 0
        total = 0
        results = []
        if pred_all is None:
            pred_all = []

        for i in range(validation_number, train_x.shape[0]-self.window_size):
            optimizer.zero_grad()
            pred_prob, _ = self(prev, train_x[i:])
            reg_weight = 1.
            if task =='regression':
                pred, _ =  self(prev, train_x[i:], prediction = True)
                loss, prev = self.lossfunc(prev, train_x[i:], y=train_x[i+1:], reg_weight = reg_weight, task = task)
            else:
                target = y[i+self.window_size-1]
                loss, prev = self.lossfunc(prev, train_x[i:], y=target, reg_weight=reg_weight, task=task)

            loss.backward()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            if task  == 'classification':
                # if i >= 200:
                #     self.prior = self.get_prior(y, i, lag=100).to(self.device)
                pred_prob = torch.mul(pred_prob, self.prior.to(self.device))

                pred_class = torch.argmax(pred_prob)
                if pred_class == target: correct += 1
                total += 1
                pred_all.append(torch.softmax(pred_prob, dim = 0).detach().cpu().numpy())
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1 - self.window_size:
                    if self.num_classes == 2:
                        auc = sklearn.metrics.roc_auc_score(y[self.window_size - 1:(i + self.window_size)].cpu().numpy(), np.asarray(pred_all)[:, 1])
                        print(i, loss.detach().cpu().numpy().item(), correct / total, pred_class, auc, pred_prob)
                        pred_prob = pred_prob.detach().cpu().numpy()
                        results.append([loss.detach().cpu().numpy().item(), correct / total, auc, pred_prob[0], pred_prob[1]])
                    else:
                        print(i, correct / total )
                        pred_prob = pred_prob.detach().cpu().numpy()
                        results.append(
                            [loss.detach().cpu().numpy().item(), correct / total, pred_prob[0], pred_prob[1]])
            elif task == 'regression':
                pred_all.append(pred.detach().cpu().numpy())
                current_mse = torch.mean((pred- train_x[i+1])**2).detach().cpu().numpy()
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                    print(i, np.mean(np.asarray(results)[:, -1]))

                results.append([loss.detach().cpu().numpy(), current_mse])

            # self.initial_bias = current_x
        return results, pred_all

    def lossfunc(self, prev, X, task, y = None, reg_weight = 1.):
        if y is None:
            conditional_likelihood, tmp = self(prev, X)
            log_likelihood = torch.mean(conditional_likelihood)
            return -log_likelihood, tmp
        else:
            if task == 'classification':
                class_weights, tmp = self(prev, X)
                class_weights = torch.mul(class_weights, self.prior.to(self.device))
                loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
                one_hot_y = one_hot(y.reshape(-1), num_classes=self.num_classes).to(self.device)
                reg = one_hot_y @ class_weights
                task_loss = loss(class_weights.reshape(1, -1), y.reshape(-1)) - reg_weight * reg
                return task_loss, tmp
            elif task == 'regression':
                pred, tmp = self(prev, X, prediction = True)
                likelihood, tmp = self(prev, X, prediction = False)
                y = y.reshape(pred.shape)
                # return torch.mean((pred - y)**2), tmp
                return - likelihood , tmp
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default=64, type=int, help='hidden states size of the model')
    parser.add_argument('--exp_data', default='movingRBF', help='dataset for the experiment')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--nc', default=2, type=int, help='number of classes')
    parser.add_argument('--mix_n', default=20, type=int, help='number of mixture components')
    parser.add_argument('--method', default='wfa', help='method to use')
    parser.add_argument('--task', default='classification', help='task to perform')
    return parser
import copy
def normalize(X):
    new_x = np.zeros(X.shape)
    tmp = copy.deepcopy(X[:1001])
    mean = np.mean(tmp, axis=0)
    std = np.std(tmp, axis=0)
    tmp = (tmp - mean) / std
    new_x[:1001] = tmp
    for i in range(1001, len(X)):
        tmp = copy.deepcopy(X[i-1000:i])
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        tmp = (X[i] - mean)/std
        new_x[i] = tmp
    #
    # tmp = X[:1000]
    # mean = np.mean(tmp, axis = 0)
    # std = np.std(tmp, axis = 0)
    return new_x

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    evaluate_interval = 100
    validation_number = 1000
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
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'Elec2')
    elif args.exp_data == 'mixeddrift':
        X, y = get_mixeddrift()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'mixedDrift')
    elif args.exp_data == 'hyperplane':
        X, y = get_hyperplane()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'hyperplane')
    elif args.exp_data == 'chess':
        X, y = get_chess()
        evaluate_interval = 1000
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'chess')
    elif args.exp_data == 'outdoor':
        X, y = get_outdoor()
        evaluate_interval = 100
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'outdoor')
    elif args.exp_data == 'covType':
        X, y = get_covtype()
        X = normalize(X[:, :10])
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 1000
    elif args.exp_data == 'rialto':
        X, y = get_rialto()
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'poker':
        X, y = get_poker()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'poker')
        evaluate_interval = 1000
    elif args.exp_data == 'interRBF':
        X, y = get_interRBF()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'rbf')
        evaluate_interval = 1000
    elif args.exp_data == 'movingRBF':
        X, y = get_movingRBF()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'rbf')
        evaluate_interval = 1000
    if args.task == 'classification':
        print(y)
        y = torch.tensor(y).type(torch.LongTensor).to(device)
        y = torch.tensor(y).reshape(-1, 1).to(device)
    X = torch.tensor(X).to(device)
    tmp_y = y.clone().cpu().numpy().reshape(-1)
    args.nc = len(Counter(tmp_y).keys())
    print(X.shape, args.nc)
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
    all_mix_ns = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    smoothing_factors = [0, 0.05, 0.1, 0.3]
    seeds = [1993]
    window_sizes = [2, 5, 7, 10]
    for r in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        for mix_n in all_mix_ns:
            for sf in smoothing_factors:
                for seed in seeds:
                    for ws in window_sizes:
                        default_parameters = {
                            'seed': seed,
                            'task': args.task,
                            'input_size': X.shape[1],
                            'encoding_size': X.shape[1],
                            'device': device,
                            'evaluate_interval': evaluate_interval,
                            'hidden_size': r,
                            'mix_n': mix_n,
                            'smoothing_factor': sf,
                            'init_std': 0.001,
                            'num_classes': args.nc,
                            'double_pre': False,
                            'initial_bias': None,
                            'window_size': ws,
                            'num_layers': 1,
                            'model': args.method
                        }
                        model = stream_density_wfa(default_parameters)

                        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
                        results, pred_all = model.fit(X[:validation_number], optimizer, y=y[:validation_number], verbose=True, scheduler=None, task=args.task)

                        if r not in validation_results:
                            validation_results[r] = {}
                            validation_results[r]['parameters'] = [default_parameters]
                            validation_results[r]['pred_all'] = [pred_all]
                            validation_results[r]['final_auc'] = [results[-1][-3]]
                        else:
                            validation_results[r]['pred_all'].append(pred_all)
                            validation_results[r]['parameters'].append(default_parameters)
                            validation_results[r]['final_auc'].append(results[-1][-3])
    # print(validation_results)
    mix_ns = {}
    max_auc = 0.
    parameters = {}
    final_r = 1
    model = None
    pred_all = None
    for r in validation_results.keys():
        for i, auc in enumerate(validation_results[r]['final_auc']):
            if auc >= max_auc:
                max_auc = auc
                parameters = validation_results[r]['parameters'][i]
                pred_all = validation_results[r]['pred_all'][i]
    model = stream_density_wfa(parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    results, pred_all = model.fit(X, optimizer, y = y, validation_number = 0, verbose=True, scheduler = None, task=args.task)
    results = np.asarray(results)
    print(results)
    save_file = {'selected_parameters': parameters, 'results': results}
    with open(os.path.join((file_dir), f'{args.method}_results'), 'wb') as f:
        pickle.dump(save_file, f)

    with open(os.path.join((file_dir), f'{args.method}_prediction'), 'wb') as f:
        pickle.dump(pred_all, f)