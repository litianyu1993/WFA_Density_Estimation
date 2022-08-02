import numpy as np
import torch
from torch import nn
import sklearn
from utils import *
from gradient_descent import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from collections import Counter
from matplotlib import pyplot as plt
from scipy import linalg as la
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
        elif self.model == 'hippo':
            q = np.arange(self.r, dtype=np.float64)
            col, row = np.meshgrid(q, q)
            r = 2 * q + 1
            M = -(np.where(row >= col, r, 0) - np.diag(q))
            T = np.sqrt(np.diag(2 * q + 1))
            self.G = T @ M @ np.linalg.inv(T)
            self.B = np.diag(T)[:, None]
            # print(self.B.shape)
            self.hippo_count = 1
            tmp_core = torch.normal(0, self.init_std, [self.r, self.d, self.r])
            self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))

            tmp_core = torch.normal(0, self.init_std, [self.r**2, self.r, self.r])
            self.out = nn.Parameter(tmp_core.clone().float().requires_grad_(True))

        # elif self.model == 'hippo':


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
        if self.model == 'lstm' or self.model == 'hippo':
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
        next = X[self.window_size]

        if self.model == 'hippo':
            (h, c) = prev
            Gt = self.G / self.hippo_count
            tmpG = la.solve_triangular(np.eye(self.r) - Gt / 2, np.eye(self.r) + Gt / 2, lower=True)
            tmpG = torch.tensor(tmpG).to(self.device).float()
            Bt = self.B / self.hippo_count
            tmpB = la.solve_triangular(np.eye(self.r) - Gt / 2, Bt, lower=True)
            tmpB = torch.tensor(tmpB).to(self.device).float()
            # print('tmpB', tmpB.shape)


            current = X[0]
            current = encoding(self, current)
            h = torch.einsum("d, i, idj -> j", current, h.ravel(), self.A).reshape(1, -1)
            h = torch.sigmoid(h)

            c = c @ tmpG + h.detach().reshape(-1) * tmpB
            tmpc = c.reshape(-1)
            # print(tmpc.shape, h.reshape(-1).shape, self.out.shape)
            h = h.reshape(-1)
            tmp = torch.einsum("d, i, dij -> j", tmpc, h, self.out).reshape(1, -1)
            # print(c.shape)

            tmp_result = phi(self, next, tmp, prediction)


            self.hippo_count += 1
            return tmp_result, (h.detach(), c)
        elif self.model == 'None':
            tmp_result = phi(self, next, prev.ravel().reshape(1, -1), prediction)
            return tmp_result, prev.detach()


        for i in range(self.window_size):

            assert self.window_size >= 1
            current = X[i]
            current = encoding(self, current)
            if self.model == 'wfa':

                tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
                tmp = torch.sigmoid(tmp)
            elif self.model == 'lstm':
                output, (state_h, state_c) = self.recurrent(X[:self.window_size].reshape([1, self.window_size, -1]),
                                                            prev)
                tmp = (state_h, state_c)


            elif self.model == 'gru':
                output, state_h = self.recurrent(X[:self.window_size].reshape([1, self.window_size, -1]), prev)
                tmp = torch.sigmoid(state_h.detach())

        if self.model == 'lstm':
            tmp_result = phi(self, next, tmp[0].reshape(1, -1), prediction)
            return tmp_result, (tmp[0].detach(), tmp[1].detach())
        else:
            tmp_result = phi(self, next, tmp.reshape(1, -1), prediction)
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
        true_label = []
        lag = self.window_size - 1
        for i in range(lag, train_x.shape[0]-self.window_size-1):
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = 0.001
            optimizer.zero_grad()
            pred_prob, _ = self(prev, train_x[i - lag:])
            reg_weight = 1.
            if task =='regression':
                pred, _ =  self(prev, train_x[i - lag:], prediction = True)
                loss, _ = self.lossfunc(prev, train_x[i - lag:], y=train_x[i+1:], reg_weight = reg_weight, task = task)
            elif task == 'classification':
                target = y[i+1]
                true_label.append(target.cpu().numpy())
                # print(i-lag, i-lag+self.window_size-1, i+1)
                loss, _ = self.lossfunc(prev, train_x[i - lag:], y=target, reg_weight=reg_weight, task=task)
            elif task == 'density':
                loss, _ = self.lossfunc(prev, train_x[i - lag:], y = train_x[i+1:], task = task)


            loss.backward()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            if task  == 'classification':
                # if i >= 200:
                #     self.prior = self.get_prior(y, i, lag=100).to(self.device)
                # pred_prob = torch.mul(pred_prob, self.prior.to(self.device))

                pred_class = torch.argmax(pred_prob).detach()
                if i >= validation_number:
                    if pred_class == target: correct += 1
                    total += 1
                pred_all.append(torch.softmax(pred_prob, dim = 0).detach().cpu().numpy())
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1 - self.window_size:
                    if i >= validation_number:
                        if self.num_classes == 2:
                            auc = sklearn.metrics.roc_auc_score(np.asarray(true_label), np.asarray(pred_all)[:, 1])
                            print(i, loss.detach().cpu().numpy().item(), correct / total, pred_class, auc, pred_prob)
                            pred_prob = pred_prob.detach().cpu().numpy()
                            results.append([loss.detach().cpu().numpy().item(), correct / total, auc, pred_prob[0], pred_prob[1]])
                        else:
                            print(i, correct / total, pred_prob[0].detach(), pred_prob[1].detach())
                            pred_prob = pred_prob.detach().cpu().numpy()
                            results.append(
                                [loss.detach().cpu().numpy().item(), correct / total, pred_prob[0], pred_prob[1]])
            elif task == 'density':
                pred_all.append(pred_prob.detach().cpu().numpy())
                if (i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1 - self.window_size:
                    print(i, loss.detach().cpu().numpy().item(), pred_prob.detach().cpu().numpy()[0])
                    results.append([loss.detach().cpu().numpy().item(), pred_prob.detach().cpu().numpy()[0], 0, 0])
            elif task == 'regression':
                pred_all.append(pred.detach().cpu().numpy())
                current_mse = torch.mean((pred- train_x[i+1])**2).detach().cpu().numpy()
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                    print(i, np.mean(np.asarray(results)[:, -1]))

                results.append([loss.detach().cpu().numpy(), current_mse])
            prev = self.transit_next_state(prev, train_x[i-lag].float())
            # self.initial_bias = train_x[i]
        return results, pred_all

    def transit_next_state(self, prev, x):
        current = x
        current = encoding(self, current)
        if self.model == 'wfa':
            tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
            tmp = torch.sigmoid(tmp).detach()
        elif self.model == 'lstm':
            output, (state_h, state_c) = self.recurrent(current.reshape([1, 1, -1]),
                                                        prev)
            tmp = (state_h.detach(), state_c.detach())

        elif self.model == 'gru':
            output, state_h = self.recurrent(current.reshape([1, 1, -1]), prev)
            tmp = state_h.detach()
            # tmp = torch.sigmoid(state_h).detach
        return tmp

    def lossfunc(self, prev, X, task, y = None, reg_weight = 1.):
        if y is None:
            conditional_likelihood, tmp = self(prev, X)
            log_likelihood = torch.mean(conditional_likelihood)
            return -log_likelihood, tmp
        else:
            if task == 'classification':
                class_weights, tmp = self(prev, X)
                # class_weights = torch.mul(class_weights, self.prior.to(self.device))
                loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
                one_hot_y = one_hot(y.reshape(-1), num_classes=self.num_classes).to(self.device)
                reg = one_hot_y @ class_weights
                task_loss = loss(class_weights.reshape(1, -1), y.reshape(-1))# - reg_weight * reg
                return task_loss, tmp
            elif task == 'density':
                likelihood, tmp = self(prev, X, prediction = False)
                return -likelihood, tmp

            elif task == 'regression':
                pred, tmp = self(prev, X, prediction = True)
                likelihood, tmp = self(prev, X, prediction = False)
                y = y.reshape(pred.shape)
                # return torch.mean((pred - y)**2), tmp
                return - likelihood , tmp
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default=64, type=int, help='hidden states size of the model')
    parser.add_argument('--exp_data', default='rialto', help='dataset for the experiment')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--nc', default=2, type=int, help='number of classes')
    parser.add_argument('--mix_n', default=20, type=int, help='number of mixture components')
    parser.add_argument('--method', default='gru', help='method to use')
    parser.add_argument('--task', default='classification', help='task to perform')
    return parser
import copy
def normalize(X):
    lag = min(int(0.2*len(X)), 1000)
    tmp = copy.deepcopy(X[:lag+1])
    new_x = np.zeros(X.shape)

    std = np.std(tmp, axis=0)
    for i in range(len(std)):
        if std[i] == 0.:
            std[i] += 0.0001
    mean = np.mean(tmp, axis=0)
    tmp = (tmp - mean) / std
    new_x[:lag+1] = tmp
    for i in range(lag+1, len(X)):
        tmp = copy.deepcopy(X[i-lag:i])
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        for j in range(len(std)):
            if std[j] == 0.:
                std[j] += 0.0001
        tmp = (X[i] - mean)/std
        new_x[i] = tmp
    #
    # tmp = X
    # mean = np.mean(tmp, axis = 0)
    # std = np.std(tmp, axis = 0)
    # new_x = (X - mean)/std
    return new_x

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    evaluate_interval = 100

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
        X = normalize(X)
        evaluate_interval = 1000
        file_dir = os.path.join(file_dir, 'artificial', 'chess')
    elif args.exp_data == 'outdoor':
        X, y = get_outdoor()
        evaluate_interval = 100
        file_dir = os.path.join(file_dir, 'realWorld', 'outdoor')
    elif args.exp_data == 'covTypetwoclasses':
        X, y = get_covtype(two_classes=True)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 1000
    elif args.exp_data == 'covType':
        X, y = get_covtype(two_classes=False)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 1000
    elif args.exp_data == 'rialtotwoclasses':
        X, y = get_rialto(two_classes=True)
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'rialto':
        X, y = get_rialto(two_classes=False)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'poker':
        X, y = get_poker(two_classes=False)
        file_dir = os.path.join(file_dir, 'realWorld', 'poker')
        evaluate_interval = 1000
    elif args.exp_data == 'pokertwoclasses':
        X, y = get_poker(two_classes=True)
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
    elif args.exp_data == 'border':
        X, y = get_border()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'border')
    elif args.exp_data == 'COIL':
        X, y = get_COIL()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'COIL')
    elif args.exp_data == 'MNIST':
        X, y = get_MNIST()
        file_dir = os.path.join(file_dir, 'datasets', 'MNIST')
    elif args.exp_data == 'gisette':
        X, y = get_gisette()
        file_dir = os.path.join(file_dir, 'datasets', 'gisette')
    elif args.exp_data == 'overlap':
        X, y = get_overlap()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'overlap')
    elif args.exp_data == 'isolet':
        X, y = get_isolet()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'isolet')

    elif args.exp_data == 'letter':
        X, y = get_letter()
        # X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'letter')

    elif args.exp_data == 'copy_paste':
        X, y = get_copy_paste(lag =10, n = 10000)
        # X = normalize(X)
        file_dir = os.path.join(file_dir, 'datasets', 'copy_paste')
        evaluate_interval = 100

    elif args.exp_data == 'hmm':
        X, ground_truth_conditionals, ground_truth_joint = get_hmm(r= 3, N = 10000)
        evaluate_interval = 100
        validation_number = 500
        # X = normalize(X, lag = 1000)
        np.savetxt('hmm_3_X.csv', X, delimiter= ',')
        np.savetxt('hmm_3_ground.csv', ground_truth_conditionals, delimiter=',')
        y = None
    validation_number = min(1000, int(len(X)*0.2))
    validation_number = len(X)
    if args.task == 'classification':
        # print(y)
        y = torch.tensor(y).type(torch.LongTensor).to(device)
        y = torch.tensor(y).reshape(-1, 1).to(device)
    X = torch.tensor(X).to(device)
    if args.task == 'classification':
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
    all_mix_ns = [2, 4, 8, 16, 32, 63, 128, 256]
    smoothing_factors = [0, 0.01, 0.05, 0.1]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    window_sizes = [1, 3, 5, 7, 9]
    for r in [2, 4, 8, 16, 32, 63, 128, 256]:
        for mix_n in all_mix_ns:
            for sf in smoothing_factors:
                for seed in seeds:
                    for ws in window_sizes:
                        print(r, mix_n, sf, seed, ws)
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
                        if args.task == 'density':
                            results, pred_all = model.fit(X[:validation_number], optimizer, y=None,
                                                          verbose=True, scheduler=None, task=args.task)
                        else:
                            results, pred_all = model.fit(X[:validation_number], optimizer, y=y[:validation_number], validation_number = 0, verbose=True, scheduler=None, task=args.task)

                        if r not in validation_results:
                            validation_results[r] = {}
                            validation_results[r]['parameters'] = []
                            validation_results[r]['pred_all'] = []
                            validation_results[r]['final_auc'] = []
                        # if r not in validation_results:
                        #     validation_results[r] = {}
                        #     validation_results[r]['parameters'] = [default_parameters]
                        #     validation_results[r]['pred_all'] = [pred_all]
                        #     if args.task == 'density':
                        #         validation_results[r]['final_auc'] = [np.mean(np.asarray(results)[:, -3])]
                        #     else:
                        #         validation_results[r]['final_auc'] = [results[-1][-3]]
                        # else:
                        validation_results[r]['pred_all'].append(pred_all)
                        validation_results[r]['parameters'].append(default_parameters)
                        if args.task == 'density':
                            validation_results[r]['final_auc'].append(np.mean(np.asarray(results)[:, -3]))
                        else:
                            validation_results[r]['final_auc'].append(results[-1][-3])
    # print(validation_results)
    mix_ns = {}
    max_auc = -9999999
    parameters = {}
    final_r = 1
    model = None
    pred_all = None

    for r in validation_results.keys():
        for i, auc in enumerate(validation_results[r]['final_auc']):
            if auc >= max_auc:
                max_auc = auc
                parameters = copy.deepcopy(validation_results[r]['parameters'][i])
                pred_all = copy.deepcopy(validation_results[r]['pred_all'][i])
    print(parameters)
    if args.task == 'density': parameters['evaluate_interval'] = 1
    model = stream_density_wfa(parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    results, pred_all = model.fit(X, optimizer, y = y, validation_number = validation_number, verbose=True, scheduler = None, task=args.task)
    results = np.asarray(results)
    # results[:, -3] += 1.
    if args.task == 'density':
        plt.plot(results[:, -3], label = 'model')
        plt.plot(ground_truth_conditionals, label = 'ground')
        plt.legend()
        plt.show()
        diff = ground_truth_conditionals[parameters['window_size']:].reshape(-1) - results[:, -3]
        print(diff.shape, ground_truth_conditionals.shape, results[:, -3].shape)
        plt.plot(diff)
        plt.show()
        new_diff = np.zeros(diff.shape)
        for i in range(1, len(new_diff)):
            new_diff[i] = np.mean(diff[:i])
        plt.plot(new_diff)
        plt.show()
        print(np.mean(diff))

    save_file = {'selected_parameters': parameters, 'results': results}
    with open(os.path.join((file_dir), f'{args.method}_results'), 'wb') as f:
        pickle.dump(save_file, f)

    with open(os.path.join((file_dir), f'{args.method}_prediction'), 'wb') as f:
        pickle.dump(pred_all, f)