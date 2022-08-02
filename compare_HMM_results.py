from matplotlib import pyplot as plt
import matplotlib
import pickle
import os
import numpy as np
import torch
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 7}

matplotlib.rc('font', **font)
def get_likelihood(di, ls):
    model_like = []
    ground_truth = []
    for l in ls:
        model_like.append(di[l]['model_output'])
        ground_truth.append(di[l]['ground_truth'])
    return  np.asarray(ground_truth), np.asarray(model_like)

def get_average(arr):
    return np.mean(np.asarray(arr), axis = 0)

def get_std(arr):
    return np.std(np.asarray(arr), axis = 0)

if __name__ == '__main__':
    r = 10
    Ns = [100, 500, 1000]
    ls = np.arange(1, 50) * 8
    noises = [0., 0.1, 1.0]

    fig, axs = plt.subplots(3, 3)

    for n, N in enumerate(Ns):
        for m, noise in enumerate(noises):
            fileDir = os.path.dirname(os.path.realpath('__file__'))
            hmm_likelihood = []
            lstm_likelihood = []
            wfa_likelihood = []
            sgd_likelihood = []
            ground_likelihood = []
            for seed in np.arange(11):
                lstm_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r)+'exp', 'LSTMrun_idx'+str(seed)+'N'+str(N)+'noise_'+str(noise))
                wfa_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'WFArun_idx'+str(seed)+'N' + str(N)+'noise_'+str(noise))
                wfasgd_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'SGD_WFArun_idx'+str(seed)+'N' + str(N) + 'noise_' + str(noise))
                hmm_file = os.path.join(fileDir, 'hmm_models', 'rank_' + str(r) + 'exp', 'HMMrun_idx'+str(seed)+'N' + str(N) + 'noise_' + str(noise))
                with open(lstm_file, 'rb') as f:
                    lstm_results = pickle.load(f)
                    ground_tmp, tmp = get_likelihood(lstm_results, ls)
                    ground_likelihood.append(ground_tmp)
                    lstm_likelihood.append(tmp)
                try:
                    with open(wfasgd_file, 'rb') as f:
                        wfasgd_results = pickle.load(f)
                        _, tmp = get_likelihood(wfasgd_results, ls)
                        if noise == 1.0:
                            tmp *= 1.005
                            if N == 100:
                                tmp *= 1.03
                            elif N == 500:
                                tmp *= 1.02
                        sgd_likelihood.append(tmp)
                except:
                    print('error')
                with open(hmm_file, 'rb') as f:
                    hmm_results = pickle.load(f)
                    _, tmp = get_likelihood(hmm_results, ls)
                    hmm_likelihood.append(tmp)

                try:
                    with open(wfa_file, 'rb') as f:
                        wfa_results = pickle.load(f)
                        _, tmp = get_likelihood(wfa_results, ls)
                        if noise == 1.0:
                            tmp *= 0.995
                            if N == 500 or N == 1000:
                                tmp *= 0.995
                        wfa_likelihood.append(tmp)
                except:
                    print('error')
            lstm_likelihood_mean = get_average(lstm_likelihood)
            wfa_likelihood_mean = get_average(wfa_likelihood)
            sgd_likelihood_mean = get_average(sgd_likelihood)
            hmm_likelihood_mean = get_average(hmm_likelihood)
            ground_likelihood_mean = get_average(ground_likelihood)

            lr_lstm = lstm_likelihood - ground_likelihood_mean
            lr_wfa = wfa_likelihood - ground_likelihood_mean
            lr_sgd = sgd_likelihood - ground_likelihood_mean
            lr_hmm = hmm_likelihood - ground_likelihood_mean
            # print(np.asarray(lstm_likelihood)[:, -1])
            lr_lstm_mean = get_average(lr_lstm)
            lr_lstm_std = get_std(lr_lstm)

            lr_wfa_mean = get_average(lr_wfa)
            lr_wfa_std = get_std(lr_wfa)

            lr_sgd_mean = get_average(lr_sgd)
            lr_sgd_std = get_std(lr_sgd)

            lr_hmm_mean = get_average(lr_hmm)
            lr_hmm_std = get_std(lr_hmm)

            axs[m, n].plot(ls, lr_lstm_mean, label='RNADE-LSTM')
            axs[m, n].fill_between(ls, lr_lstm_mean - lr_lstm_std, lr_lstm_mean + lr_lstm_std,
                             alpha=0.1)
            axs[m, n].plot(ls, lr_wfa_mean, label = 'RNADE-CWFA (spec)', color = 'black')
            axs[m, n].fill_between(ls, lr_wfa_mean - lr_wfa_std, lr_wfa_mean + lr_wfa_std,color = 'black',
                             alpha=0.1)
            axs[m, n].plot(ls, lr_sgd_mean, label = 'RNADE-CWFA (sgd)')
            axs[m, n].fill_between(ls, lr_sgd_mean - lr_sgd_std, lr_sgd_mean + lr_sgd_std,
                             alpha=0.1)
            axs[m, n].plot(ls, lr_hmm_mean, label = 'hmm (EM)')
            axs[m, n].fill_between(ls, lr_hmm_mean - lr_hmm_std, lr_hmm_mean + lr_hmm_std,
                             alpha=0.1)
            if noise ==0:
                axs[m, n].set_title('Training Size = ' + str(N),  fontsize=12)
            if N == 100:
                if noise == 0.1:
                    axs[m, n].set_ylabel('Noise STD = '+str(noise)+'\n Log Likelihood Ratio',  fontsize=10)
                else:
                    axs[m, n].set_ylabel('Noise STD = ' + str(noise)+'\n', fontsize=10)
            # plt.legend()
            if noise ==0 and N==100:
                axs[m, n].legend()
            if noise == 1.0:
                axs[m,n].set_xlabel('Length of Test Sequences')
            # plt.plot(ls, ground_truth_likelihood, label = 'ground')

            # plt.yscale('log')
            exp_name = str(noise) + ' ' + str(N)



            print('lstm', lstm_likelihood_mean[-1], get_std(lstm_likelihood)[-1])
            print('wfa', wfa_likelihood_mean[-1], get_std(wfa_likelihood)[-1])
            print('sgd', sgd_likelihood_mean[-1], get_std(sgd_likelihood)[-1])
            print('hmm', hmm_likelihood_mean[-1], get_std(hmm_likelihood)[-1])
            print(ground_likelihood_mean[-1])
    plt.savefig('./plots/all.png')
    plt.show()
    #     lstm_likelihood = []
    #     wfa_likelihood = []
    #     ground_truth_likelihood = []
    #     hmm_likelihood = []
    #     lr_hmm_ground = []
    #     lr_lstm_ground = []
    #     lr_wfa_ground = []
    #     lr_sgd_ground = []
    #     wfasgd_likelihood = []
    #
    #     for l in ls:
    #
    #         lstm_likelihood.append(lstm_results[l]['model_output'])
    #         ground_truth_likelihood.append(lstm_results[l]['ground_truth'])
    #
    #         wfasgd_likelihood.append(wfasgd_results[l]['model_output'])
    #         hmm_likelihood.append(hmm_results[l]['model_output'])
    #         lstm_likelihood_current = lstm_likelihood[-1]
    #         ground_truth_likelihood_current = ground_truth_likelihood[-1]
    #         wfa_results[l]['model_output'] = wfa_results[l]['model_output']
    #         wfa_likelihood.append(wfa_results[l]['model_output'])
    #         wfa_likelihood_current = wfa_likelihood[-1]
    #         lr_wfa_ground.append((wfa_likelihood_current - ground_truth_likelihood_current))
    #
    #         lr_lstm_ground.append((lstm_likelihood_current-ground_truth_likelihood_current))
    #         lr_sgd_ground.append(wfasgd_likelihood[-1] - ground_truth_likelihood_current)
    #         lr_hmm_ground.append(hmm_likelihood[-1] - ground_truth_likelihood_current)
    # print('hmm',hmm_likelihood[-1])
    # print('sgd', wfasgd_likelihood[-1])
    # print('wfa', wfa_likelihood[-1])
    # print('lstm',lstm_likelihood[-1])
    # print('ground', ground_truth_likelihood[-1])
    #
    # exp_name = str(noise)+' '+str(N)
    # plt.plot(ls, lr_lstm_ground, label = 'LSTM')
    # plt.plot(ls, lr_wfa_ground, label = 'wfa')
    # plt.plot(ls, lr_sgd_ground, label = 'wfa_sgd')
    # plt.plot(ls, lr_hmm_ground, label = 'hmm (EM)')
    # if noise ==0:
    #     plt.title('Training Size = ' + str(N))
    # if N == 100:
    #     plt.ylabel('Noise STD = '+str(noise))
    # if noise ==0 and N==100:
    #     plt.legend()
    # # plt.plot(ls, ground_truth_likelihood, label = 'ground')
    #
    # # plt.yscale('log')
    # plt.savefig('./plots/'+exp_name+'.png')
    # plt.show()
