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
        if parameter_dict['prob'] == 'yes':
            self.prob = True
        else:
            self.prob = False

        # self.encoder_batchnorm = nn.BatchNorm1d(self.xd)
        # self.hidden_batchnorm = nn.BatchNorm1d(self.r)


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
        elif self.model == 'no_rec':
            tmp_core = torch.normal(0, 1, [self.d, self.r])
            self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
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
            self.no_prob_class_out = torch.nn.Linear(self.r,  self.num_classes, bias=True).requires_grad_(True)
            self.mu_outs = torch.nn.Linear(self.r, self.mix_n * self.xd, bias=True).requires_grad_(True)
            self.mu_outs2 = torch.nn.Linear(self.mix_n * self.xd,  self.mix_n *  self.xd * self.num_classes, bias=True).requires_grad_(True)
            self.sig_outs = torch.nn.Linear( self.r,  self.mix_n *  self.xd * self.num_classes,  bias=True).requires_grad_(True)
            self.alpha_outs = torch.nn.Linear( self.r,  self.mix_n * self.num_classes, bias=True).requires_grad_(True)
            # for i in range(self.num_classes):
            #     self.mu_outs.append(torch.nn.Linear(self.r, self.mix_n * self.xd, bias=True).requires_grad_(True))
            #     self.mu_outs2.append(torch.nn.Linear(self.mix_n * self.xd,  self.mix_n *  self.xd, bias=True).requires_grad_(True))
            #     self.sig_outs.append(torch.nn.Linear( self.r,  self.mix_n *  self.xd, bias=True).requires_grad_(True))
            #     self.alpha_outs.append(torch.nn.Linear( self.r,  self.mix_n, bias=True).requires_grad_(True))
        else:
            self.num_classes = None
            self.mu_out = torch.nn.Linear(self.r , self.mix_n * self.xd, bias=True).requires_grad_(True)
            self.sig_out = torch.nn.Linear(self.r , self.mix_n * self.xd, bias=True).requires_grad_(True)

            self.alpha_out = torch.nn.Linear(self.r, self.r, bias=True).requires_grad_(True)
            self.alpha_out2 = torch.nn.Linear(self.r, self.mix_n, bias=True).requires_grad_(True)
            self.mu_out2 = torch.nn.Linear( self.mix_n *  self.xd,  self.mix_n * self.xd, bias=True).requires_grad_(True)
            self.sig_out2 = torch.nn.Linear(self.mix_n * self.xd, self.mix_n * self.xd, bias=True).requires_grad_(True)
            self.mu_out3 = torch.nn.Linear(self.mix_n * self.xd, self.mix_n * self.xd, bias=True).requires_grad_(True)
            self.sig_out3 = torch.nn.Linear(self.mix_n * self.xd, self.mix_n * self.xd, bias=True).requires_grad_(True)

            self.mu_bias = nn.Parameter(torch.normal(0, self.init_std, [self.mix_n * self.xd])).requires_grad_(True)
            sig1 = torch.normal(-2, self.init_std, [int(self.mix_n / 2)])
            sig2 = torch.normal(0, self.init_std, [self.mix_n - int(self.mix_n / 2)])
            sig = torch.cat([sig1, sig2], dim=0)
            self.sig_bias = nn.Parameter(sig.reshape(self.mix_n * self.xd)).requires_grad_(True)

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


    def forward(self, prev, X, prediction = False, loss = False):
        if self.double_pre:
            X = X.double().to(device)
        else:
            X = X.float().to(device)

        all_results = []
        next = X[self.window_size]
        # next = torch.ones(next.shape).to(self.device)

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
            if not self.prob:
                current = X[i+1]
            else:
                current = X[i]
            current = encoding(self, current)
            all_liklihood = 0.
            if self.model == 'wfa':

                tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
                tmp = torch.sigmoid(tmp)

                # if loss:
                #     next_x = X[i + 1]
                #     tmp_result, gmm_params = phi(self, next_x, tmp.reshape(1, -1), prediction)
                #     all_liklihood += tmp_result
            elif self.model == 'no_rec':

                tmp = torch.einsum("d, dj -> j", current, self.A).reshape(1, -1)
                tmp = torch.sigmoid(tmp)
            elif self.model == 'lstm':
                output, (state_h, state_c) = self.recurrent(current.reshape([1, 1, -1]),
                                                            prev)
                tmp = (state_h, state_c)
                # next_x = X[i+1]
                # out_tmp = torch.cat([tmp[0].reshape(1, -1), next_x.reshape(1, -1)], dim=1)


            elif self.model == 'gru':
                output, state_h = self.recurrent(current.reshape([1, 1, -1]), prev)
                tmp =(state_h)
            prev = tmp
        # print(tmp)
        # if loss:
        #     return all_liklihood, (tmp[0].detach(), tmp[1].detach()), gmm_params
        # print('next x is:', next, 'current x is:', X[self.window_size-1])
        if self.model == 'lstm':
            tmp_result, gmm_params = phi(self, next, tmp[0].reshape(1, -1), prediction)
            return tmp_result, (tmp[0].detach(), tmp[1].detach()), gmm_params
        else:
            tmp_result, gmm_params = phi(self, next, tmp.reshape(1, -1), prediction)
            return tmp_result, tmp.detach(), gmm_params

    def fit_density(self, train_x, optimizer, y = None, verbose = True, scheduler = None, pred_all = None, validation_number = 0, task = 'classification', ground_conditionals = None):
        if self.model == 'wfa' or self.model == 'no_rec':
            prev = torch.softmax(torch.rand([1, self.A.shape[0]]), dim = 1).to(device)
        else:
            prev = self.init_state(1)
        lag = self.window_size - 1
        for i in range(lag, train_x.shape[0] - self.window_size - 1):
            MISE_tmp = []
            L_inf_tmp = []
            for j in range(train_x.shape[1]):
                optimizer.zero_grad()
                pred_prob, _, gmm_params = self(prev, train_x[i - lag:, i])
                loss, _ = self.lossfunc(prev, train_x[i - lag:, i], y=train_x[i + 1:, i], task=task)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss)
                if task == 'density':
                    pred_all.append(pred_prob.detach().cpu().numpy())
                    MISE = self.MISE(ground_conditionals, train_x, i, gmm_params)
                    L_inf = self.L_inf(ground_conditionals, train_x, i, gmm_params)
                    MISE_tmp.append(MISE)
                    L_inf_tmp.append(L_inf)
            MISE = np.mean(np.asarray(MISE_tmp))
            L_inf = np.mean(np.asarray(L_inf_tmp))

            if (i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1 - self.window_size:
                # MISE = self.MISE(ground_conditionals, sampled_x, i, gmm_params)
                # L_inf = self.L_inf(ground_conditionals, sampled_x, i, gmm_params)
                print(i, loss.detach().cpu().numpy().item(), MISE, L_inf)
                results.append([loss.detach().cpu().numpy().item(), MISE, L_inf, 0, 0])

    def fit(self,train_x, optimizer, y = None, verbose = True, scheduler = None, pred_all = None, validation_number = 0, task = 'classification', ground_conditionals = None, sampled_x = None):
        if self.model == 'wfa' or self.model == 'no_rec':
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
            pred_prob, _, gmm_params = self(prev, train_x[i - lag:])
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
                loss, _ = self.lossfunc(prev, train_x[i - lag:], y = train_x[i - lag:], task = task)


            loss.backward()
            # for name, param in self.named_parameters():
            #     # if param.requires_grad:
            #     print(name, param.data)
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
                pred_prob = pred_prob.reshape(-1, )
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
                self.evaluate_interval = 100
                if (i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1 - self.window_size:
                    MISE = self.MISE(ground_conditionals, sampled_x, i+self.window_size-1, gmm_params)
                    L_inf = self.L_inf(ground_conditionals, sampled_x, i, gmm_params)
                    print(i, loss.detach().cpu().numpy().item(), MISE, L_inf)
                    results.append([loss.detach().cpu().numpy().item(), MISE, L_inf, 0, 0])
            elif task == 'regression':
                pred_all.append(pred.detach().cpu().numpy())
                current_mse = torch.mean((pred- train_x[i+1])**2).detach().cpu().numpy()
                if (i >= 100 and i % self.evaluate_interval == 0) or i == train_x.shape[0] - 1:
                    print(i, np.mean(np.asarray(results)[:, -1]))

                results.append([loss.detach().cpu().numpy(), current_mse])
            prev = self.transit_next_state(prev, train_x[i-lag].float())
            # self.initial_bias = train_x[i]
        self.prev = prev
        return results, pred_all

    def transit_next_state(self, prev, x):
        current = x
        current = encoding(self, current)
        if self.model == 'wfa':
            tmp = torch.einsum("d, i, idj -> j", current, prev.ravel(), self.A).reshape(1, -1)
            tmp = torch.sigmoid(tmp).detach()
        elif self.model == 'no_rec':

            tmp = torch.einsum("d, dj -> j", current, self.A).reshape(1, -1)
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

    def MISE(self, ground, sampled_X, index, gmm_params):
        ave = 0.
        mu, sig, alpha = gmm_params[0], gmm_params[1], gmm_params[2]
        all_pred = []
        if len(sampled_X) <= index: return 0.
        for i in range(len(sampled_X[index])):
            x = sampled_X[index][i].reshape(-1, 1)
            pred = torch_mixture_gaussian(torch.tensor(x).to(self.device), mu, sig, alpha, prediction = False)
            pred = pred.detach().cpu().numpy()[0]
            pred = np.exp(pred)
            tmp_ground = np.exp(ground[index][i])
            ave += (pred - tmp_ground)**2
            all_pred.append(pred)
        std = np.std(np.asarray(all_pred))
        return ave/len(sampled_X[index]) + std

    def L_inf(self, ground, sampled_X, index, gmm_params):
        ave = 0.
        mu, sig, alpha = gmm_params[0], gmm_params[1], gmm_params[2]
        diff = []
        if len(sampled_X) <= index: return 0.
        for i in range(len(sampled_X[index])):
            x = sampled_X[index][i].reshape(-1, 1)
            pred = torch_mixture_gaussian(torch.tensor(x).to(self.device), mu, sig, alpha, prediction=False)
            pred = pred.detach().cpu().numpy()[0]
            pred = np.exp(pred)
            tmp_ground = np.exp(ground[index][i])
            diff.append(np.abs(pred - tmp_ground))
        return np.max(np.asarray(diff))

    def lossfunc(self, prev, X, task, y = None, reg_weight = 1.):
        if y is None:
            conditional_likelihood, tmp = self(prev, X)
            log_likelihood = torch.mean(conditional_likelihood)
            return -log_likelihood, tmp
        else:
            if task == 'classification':
                class_weights, tmp, _ = self(prev, X)
                loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
                task_loss = loss(class_weights.reshape(1, -1), y.reshape(-1))# - reg_weight * reg
                return task_loss, tmp
            elif task == 'density':
                total_likelihood = 0.
                likelihood, tmp, gmm_params_tmp = self(prev, X, prediction=False)
                pred, tmp, _ = self(prev, X, prediction=True)
                # for i in range(15):
                #     likelihood, tmp, gmm_params_tmp = self(prev, X[i:], prediction = False)
                #     total_likelihood += likelihood
                # mu = gmm_params_tmp[0]
                sig = gmm_params_tmp[1]
                # alpha = gmm_params_tmp[-1]
                # l2_loss = torch.min((mu[0, :, :] - X[self.window_size])**2)
                reg_var = torch.mean(sig[0, :, :])
                # print(sig[0])
                # mean = (torch.abs(likelihood.detach()) + reg_var.detach())/2
                return -likelihood,  tmp

            elif task == 'regression':
                pred, tmp, _ = self(prev, X, prediction = True)
                likelihood, tmp, _ = self(prev, X, prediction = False)
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
    parser.add_argument('--window_size', default=1, type=int, help='size of sliding window')
    parser.add_argument('--method', default='gru', help='method to use')
    parser.add_argument('--task', default='classification', help='task to perform')
    parser.add_argument('--N', default=2000, type = int, help='number of examples')
    parser.add_argument('--prob', default='yes', help='if using probability')
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
    ground_conditionals = None
    sampled_x = None
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
        evaluate_interval = 950
        file_dir = os.path.join(file_dir, 'artificial', 'chess')
    elif args.exp_data == 'outdoor':
        X, y = get_outdoor()
        evaluate_interval = 100
        file_dir = os.path.join(file_dir, 'realWorld', 'outdoor')
    elif args.exp_data == 'covTypetwoclasses':
        X, y = get_covtype(two_classes=True)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 950
    elif args.exp_data == 'covType':
        X, y = get_covtype(two_classes=False)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'covType')
        evaluate_interval = 950
    elif args.exp_data == 'rialtotwoclasses':
        X, y = get_rialto(two_classes=True)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'rialto':
        X, y = get_rialto(two_classes=False)
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'realWorld', 'rialto')
    elif args.exp_data == 'poker':
        X, y = get_poker(two_classes=False)
        file_dir = os.path.join(file_dir, 'realWorld', 'poker')
        evaluate_interval = 950
    elif args.exp_data == 'pokertwoclasses':
        X, y = get_poker(two_classes=True)
        file_dir = os.path.join(file_dir, 'realWorld', 'poker')
        evaluate_interval = 1000
    elif args.exp_data == 'interRBF':
        X, y = get_interRBF()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'interrbf')
        evaluate_interval = 950
    elif args.exp_data == 'movingRBF':
        X, y = get_movingRBF()
        X = normalize(X)
        file_dir = os.path.join(file_dir, 'artificial', 'movingrbf')
        evaluate_interval = 950
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

        X, ground_truth_conditionals = get_hmm(r= 3, N = args.N, n = 10)
        evaluate_interval = 100
        validation_number = 500
        X = X[0]

        sampled_X = copy.deepcopy(X).reshape(len(X), -1)
        X = X[:, 0].reshape(-1, 1)
        ground_truth_conditionals = ground_truth_conditionals[0]
        # X = normalize(X)
        print(ground_truth_conditionals.shape, X.shape, sampled_X.shape)
        np.savetxt('hmm_3_X.csv', X, delimiter= ',')
        np.savetxt('hmm_3_ground.csv', ground_truth_conditionals, delimiter=',')

        y = None
    else:
        X, ground_truth_conditionals = get_artificial_distribution(density_name = args.exp_data, N=args.N, sample_n = 10)
        evaluate_interval = 100
        validation_number = 500
        sampled_X = copy.deepcopy(X)
        # for i in range(sampled_X.shape[1]):
        #     sampled_X[:, i] = normalize(sampled_X[:, i].reshape(-1, 1)).reshape(-1,)
        X = X[:, 0].reshape(-1, 1)
        # X = (X - np.mean(X, axis = 0))/ np.std(X, axis = 0)
        # X = normalize(X)
        print(ground_truth_conditionals.shape, X.shape, sampled_X.shape)
        np.savetxt('hmm_3_X.csv', X, delimiter=',')
        np.savetxt('hmm_3_ground.csv', ground_truth_conditionals, delimiter=',')

        y = None

    if  not os.path.exists(file_dir):
        os.makedirs(file_dir)
    validation_number = min(1000, int(len(X)*0.2))
    # validation_number = len(X)
    if args.task == 'classification':
        # print(y)
        y = torch.tensor(y).type(torch.LongTensor).to(device)
        y = torch.tensor(y).reshape(-1, 1).to(device)
    X = torch.tensor(X).to(device)
    if args.task == 'classification':
        tmp_y = y.clone().cpu().numpy().reshape(-1)
        args.nc = len(Counter(tmp_y).keys())
    print(X.shape, args.nc, args.lr)
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
    all_mix_ns = [args.mix_n]
    smoothing_factors = [0]
    seeds = [1993]
    # if args.method == 'no_rec':
    #     window_sizes = [1]
    # else:
    window_sizes = [args.window_size]
    for r in [args.r]:
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
                            'model': args.method,
                            'prob': args.prob
                        }
                        model = stream_density_wfa(default_parameters)

                        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
                        if args.task == 'density':
                            results, pred_all = model.fit(X[:validation_number], optimizer, y=None,
                                                          verbose=True, scheduler=None, task=args.task, ground_conditionals=ground_truth_conditionals, sampled_x=sampled_X)
                        else:

                            results, pred_all = model.fit(X[:validation_number], optimizer, y=y[:validation_number], validation_number = 0, verbose=True, scheduler=None, task=args.task)

                        if r not in validation_results:
                            validation_results[r] = {}
                            validation_results[r]['parameters'] = []
                            validation_results[r]['pred_all'] = []
                            validation_results[r]['final_auc'] = []
                        validation_results[r]['pred_all'].append(pred_all)
                        validation_results[r]['parameters'].append(default_parameters)
                        if args.task == 'density':
                            validation_results[r]['final_auc'].append(np.mean(np.asarray(results)[:, -3]))
                        else:
                            validation_results[r]['final_auc'].append(results[-1][-3])

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
    if args.task == 'density':
        results, pred_all = model.fit(X, optimizer, y = y, validation_number = validation_number, verbose=True, scheduler = None, task=args.task, ground_conditionals=ground_truth_conditionals, sampled_x=sampled_X)
    else:
        results, pred_all = model.fit(X, optimizer, y=y, validation_number=validation_number, verbose=True,
                                      scheduler=None, task=args.task, ground_conditionals=None,
                                      sampled_x=None)
    results = np.asarray(results)
    if args.task == 'density':
        # plt.plot(results[:, -3], label = args.method)
        # plt.legend()
        # plt.show()
        #
        # plt.plot(results[:, -4], label=args.method)
        # plt.legend()
        # plt.show()
        print('L_inf', 'MISE')
        print(np.mean(results[:, -3]), np.mean(results[:, -4]))
        pred_prob, _, gmm_params = model(model.prev, X[-model.window_size-1:])
        ave = 0.
        mu, sig, alpha = gmm_params[0], gmm_params[1], gmm_params[2]
        exp_pred = []
        exp_ground = []
        # sampled_X, ground_truth_conditionals = get_artificial_distribution(density_name=args.exp_data, N=2000, sample_n=10)

        for j in range(len(sampled_X)):
            for i in range(len(sampled_X[-1])):
                x = sampled_X[j][i].reshape(-1, 1)
                pred = torch_mixture_gaussian(torch.tensor(x).to(model.device), mu, sig, alpha, prediction=False)
                pred = pred.detach().cpu().numpy()[0]
                pred = np.exp(pred)
                tmp_ground = np.exp(ground_truth_conditionals[j][i])
                exp_pred.append(pred)
                exp_ground.append(tmp_ground)

        plt.scatter(sampled_X.reshape(-1, ), exp_pred, label = args.method)
        plt.scatter(sampled_X.reshape(-1, ), exp_ground, label = 'ground')
        plt.legend()
        plt.show()


        # new_diff = np.zeros(len(results[:, -3]))
        # for i in range(100, len(results[:, -3])):
        #     new_diff[i] = np.mean(results[:, -3][i-100:i])
        # plt.plot(new_diff[100:])
        # plt.show()


    save_file = {'selected_parameters': parameters, 'results': results}
    with open(os.path.join((file_dir), f'{args.method}_{args.exp_data}_with_prob_{args.prob}_results'), 'wb') as f:
        pickle.dump(save_file, f)