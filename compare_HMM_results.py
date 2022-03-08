from matplotlib import pyplot as plt
import pickle
import os
import numpy as np
import torch

if __name__ == '__main__':
    r = 20
    N = 100
    ls = np.arange(2, 1000)
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    lstm_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r)+'exp', 'LSTMrun_idx0N'+str(N))
    wfa_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'WFArun_idx0N' + str(N))
    with open(lstm_file, 'rb') as f:
        lstm_results = pickle.load(f)
    with open(wfa_file, 'rb') as f:
        wfa_results = pickle.load(f)

    lstm_likelihood = []
    wfa_likelihood = []
    ground_truth_likelihood = []
    for l in ls:
        lstm_likelihood.append(-lstm_results[l]['model_output'].detach().cpu().numpy() + lstm_results[l]['ground_truth'])
        ground_truth_likelihood.append(-lstm_results[l]['ground_truth'])
        wfa_likelihood.append(-wfa_results[l]['model_output'].detach().cpu().numpy() + lstm_results[l]['ground_truth'])

    plt.plot(lstm_likelihood, label = 'LSTM')
    plt.plot(wfa_likelihood, label = 'wfa')
    # plt.plot(ground_truth_likelihood, label = 'ground')
    plt.legend()
    # plt.yscale('log')
    plt.show()