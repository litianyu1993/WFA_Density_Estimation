import numpy as np

from stream_sgd_wfa import *
from scipy.signal import savgol_filter
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
        if self.model== 'lstm':
            return (torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float(),
                    torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float())
        else:
            return torch.zeros(self.RNN_num_layers, N, self.RNN_hidden_size).to(device).float()
    def lossfunc(self, prev, current, y = None):
        if y is None:
            log_likelihood, _ = self(prev, current)
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
            loss = self.lossfunc(prev, current_x)
            loss.backward()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            results.append([joint_likelihood, -loss.detach().cpu().numpy()])
            print(i, results[-1])
        return results, pred_all

class stream_density_wfa(nn.Module):

    def __init__(self, xd, d, r, mix_n, device, evaluate_interval = 500, num_classes = None, use_batchnorm = True, init_std = 0.001, double_pre = False, initial_bias = None, smoothing_factor = 0.):
        super().__init__()
        self.xd= xd
        self.d = d
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.encoder_1 = torch.nn.Linear(xd, d, bias=True)
        self.evaluate_interval = evaluate_interval
        self.r = r
        self.mix_n = mix_n
        self.smoothing_factor = smoothing_factor

        tmp_core = torch.normal(0, init_std, [r, d, r])
        self.A = nn.Parameter(tmp_core.clone().float().requires_grad_(True))
        self.num_classes = num_classes

        self.mu_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.sig_out = torch.nn.Linear(r, mix_n * xd, bias=True).requires_grad_(True)
        self.alpha_out = torch.nn.Linear(r, mix_n, bias=True).requires_grad_(True)
        self.mu_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)
        self.sig_out2 = torch.nn.Linear(mix_n * xd, mix_n * xd, bias=True).requires_grad_(True)

        self.batchnrom = nn.BatchNorm1d(self.A.shape[-1])
        self.double_pre = double_pre
        self.initial_bias = initial_bias
        if double_pre:
            self.double().to(device)
        else:
            self.float().to(device)


    def forward(self, prev, current, prediction = False):
        if self.double_pre:
            prev = prev.double().to(device)
            current = current.double().to(device)
        else:
            prev = prev.float().to(device)
            current = current.float().to(device)
        tran = self.A
        # tmp = self.encoder_1(current.reshape(1, -1))
        # print(current)
        tmp_result = phi(self, current, prev, prediction, use_relu=True)
        tmp = torch.einsum("d, i, idj -> j", current.ravel(), prev.ravel(), tran).reshape(1, -1)
        return tmp_result, tmp

    def fit(self, train_x, optimizer, y=None, verbose=True, scheduler=None, pred_all=None, validation_number=0):
        prev = torch.softmax(torch.rand([1, self.A.shape[0]]), dim=1).to(device)
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
            loss, prev = self.lossfunc(prev, current_x)

            prev = torch.sigmoid(prev.detach())
            loss.backward()
            joint_likelihood += -loss.detach().cpu().numpy()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            results.append([joint_likelihood, -loss.detach().cpu().numpy()])
            # print(i, results[-1][1])
            # self.initial_bias = current_x

        return results, pred_all

    def eval_likelihood(self, X, batch = False):
        log_likelihood, hidden_norm = self(X)
        log_likelihood = log_likelihood.detach().cpu().numpy()
        if not batch:
            return np.mean(log_likelihood)
        else:
            return log_likelihood

    def lossfunc(self, prev, current, y = None, prior_1 = None):
        if y is None:
            conditional_likelihood, tmp = self(prev, current)
            log_likelihood = torch.mean(conditional_likelihood)
            return -log_likelihood, tmp
        else:
            class_weights, tmp = self(prev, current)
            loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
            y = y.reshape(-1)
            if prior_1 is not None:
                class_weights = class_weights + torch.log(torch.tensor([1-prior_1, prior_1])).to(device)
            return loss(class_weights.reshape(1, -1), y), tmp

    def eval_prediction(self, X, y):
        if self.double_pre:
            y = y.double().to(device)
        else:
            y = y.float().to(device)
        pred_mu, pred_sig = self(X, prediction = True)
        pred_mu = pred_mu.reshape(y.shape)
        print(pred_mu[0], y[0])
        return MAPE(pred_mu, y), torch.mean(pred_sig), torch.mean((pred_mu - y)**2), torch.mean(y**2), torch.mean(pred_mu**2)

def compute_hmmlearn_conditionals(hmmmodel, X):
    scores = []
    for i in range(1, len(X)):
        scores.append(hmmmodel.score(X[:i]))
    scores = np.asarray(scores)
    scores = scores[1:] - scores[:-1]
    return scores

if __name__ == '__main__':

    nshmm = incremental_HMM(r=10, seed = 1993)

    # train_x, test_x, nshmm = ggh.gen_gmmhmm_data(N=1, xd=1, L=1, r=20, seed= 1993)

    # plt.plot(ground_truth_conditionals, label=ground_truth_conditionals)
    all_results = []
    for method in ['wfa']:
        for r in [2]:
            for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
                np.random.seed(seed)
                # X, _ = nshmm.sample(1000)
                X = nshmm.sample(1000, seed)
                X = np.swapaxes(X, 0, 1)


                # ground_truth_conditionals = compute_hmmlearn_conditionals(nshmm, X)
                # print(ground_truth_conditionals)
                ground_truth_joint, ground_truth_conditionals = nshmm.score(X, stream=True)
                ground_truth_conditionals = np.asarray(ground_truth_conditionals).reshape(-1)
                ground_truth_joint = np.asarray(ground_truth_joint)
                # X = np.insert(X, X.shape[1], 1, axis=1)

                X = torch.tensor(X).to(device)
                X = X.reshape(X.shape[0], -1)
                current_results = {'r':r, 'method': method, 'seed': seed, 'ground': ground_truth_conditionals}
                lr = 0.01
                if method == 'lstm' or method == 'gru':
                    default_parameters = {
                        'input_size': X.shape[-1],
                        'RNN_hidden_size': r,
                        'RNN_num_layers': 1,
                        'device': device,
                        'num_classes': None,
                        'mixture_number': r,
                        'model': method,
                        'evaluate_interval': 1,
                        'smoothing_factor': 0.
                    }
                    #
                    model = stream_RNADE_RNN(default_parameters)
                elif method == 'wfa':
                    model = stream_density_wfa(xd=X.shape[1], num_classes=None, d=X.shape[1], r=r, mix_n=r, device=device,
                                               initial_bias=None, evaluate_interval=1, smoothing_factor=0.)
                optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
                # optimizer = optim.SGD(model.parameters(), lr = lr)
                try:
                    results, pred_all = np.asarray(model.fit(X, optimizer, verbose=True, scheduler=None))
                    results = np.asarray(results)
                    print(r, results[-1], ground_truth_conditionals[-1], ground_truth_joint[-1])
                    current_results['model_output'] = results[:, -1]
                    all_results.append(current_results)
                except:
                    print('failed')
                # all_results.append(.reshape(all_results[-1].shape))
                # print(all_results[-1].shape)
                # plt.plot(savgol_filter(results[:, -1], 51, 4), label = f'{method}_hidden_size{r}')
                # plt.plot(results[:, -1], label=f'{method}_hidden_size{r}')
        # plt.legend()
        # plt.show()
    # print(np.asarray(all_results).shape)
    with open('exp_drift_hmm.results_ranks_5', 'wb') as f:
        pickle.dump(all_results, f)
    # np.savetxt('exp_drift_hmm.results', all_results, delimiter=',' )

