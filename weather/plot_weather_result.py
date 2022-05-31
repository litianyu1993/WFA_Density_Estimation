import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

if __name__ == '__main__':

    results = {}
    test_l = np.arange(1, 20)*8
    for n in [200, 1000, 10000]:
        for rank in [10, 20, 30]:
            for l in [5]:
                for method in ['LSTM', 'Transformer', 'SGD_WFA']:
                    tmp_results = []
                    if method == 'SGD_WFA' and n == 10000 : n = 15000
                    if method == 'SGD_WFA': rank = 20
                    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        try:
                            file_dir = os.path.join('rank_'+str(rank), method+'run_idx'+str(seed)+'N'+str(n)+'noise_0.0L_'+str(l)+'results')
                            if not os.path.exists(file_dir): continue
                            with open(file_dir, 'rb') as f:
                                result = pickle.load(f)
                            test_result = []
                            for tl in test_l:
                                test_result.append(result[tl]['model_output'])
                            tmp_results.append(test_result)

                        except:
                            print('not finished')

                        # print(test_result)
                    tmp_results = np.asarray(tmp_results)
                    print(tmp_results, method, n)
                    std = np.std(tmp_results, axis=0)
                    ave_results = np.mean(tmp_results, axis = 1)
                    index = np.argmax(ave_results)
                    tmp_results = tmp_results[index]

                    # tmp_results = np.max(tmp_results, axis = 0)
                    if method not in results.keys():
                        results[method] = {}
                    if n not in results[method].keys():
                        results[method][n] = {}
                    if rank not in results[method][n].keys():
                        results[method][n][rank] = {}
                    if l not in results[method][n][rank].keys():
                        results[method][n][rank][l] = [tmp_results, std]
                    results[method][n][rank][l] = [tmp_results, std]
                    if n == 15000: n = 10000

    for n in [200, 1000, 10000]:
        tmp_mean = []
        tmp_std = []
        for rank in[10, 20, 30]:

            lstm_mean = results['LSTM'][n][rank][5][0]
            lstm_std = results['LSTM'][n][rank][5][1]

            transformer_mean = results['Transformer'][n][rank][5][0]
            transformer_std = results['Transformer'][n][rank][5][0]
            if n ==10000: N = 15000
            else:
                N = n
            wfa_mean = results['SGD_WFA'][N][20][5][0]
            wfa_std = results['SGD_WFA'][N][20][5][1]

            plt.plot(test_l, lstm_mean, label = 'RNADE-LSTM')
            plt.fill_between(test_l, lstm_mean - lstm_std, lstm_mean + lstm_std,
                          alpha=0.1)

            plt.plot(test_l, transformer_mean, label='RNADE-transformer')
            plt.fill_between(test_l, transformer_mean - transformer_std, transformer_mean + transformer_std,
                             alpha=0.1)

            plt.plot(test_l, wfa_mean, label='RNADE-WFA')
            plt.fill_between(test_l, wfa_mean - wfa_std, wfa_mean + wfa_std,
                             alpha=0.1)
            plt.legend()
            plt.title(f"num {n}, rank {rank}")
            plt.show()
            tmp_mean.append([lstm_mean[-1], transformer_mean[-1], wfa_mean[-1]])
            tmp_std.append([lstm_std[-1], transformer_std[-1], wfa_std[-1]])

        tmp_mean = np.asarray(tmp_mean)
        tmp_std = np.asarray(tmp_std)
        r1 = np.arange(len(tmp_mean[:, 0]))
        barWidth = 0.2
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]
        plt.bar(r1, tmp_mean[:, 0], width=barWidth,  edgecolor='black', yerr= 0, capsize=7, label='RNADE_LSTM')
        plt.bar(r2, tmp_mean[:, 1], width=barWidth, edgecolor='black', yerr=0, capsize=7,
                label='RNADE-transformer')
        plt.bar(r3, tmp_mean[:, 2], width=barWidth, edgecolor='black', yerr=0, capsize=7,
                label='RNADE-WFA')
        plt.bar(r4, -3645, width=barWidth, edgecolor='black', yerr=0, capsize=7,
                label='Flow-baseline')
        # general layout
        plt.xticks([r + barWidth for r in range(len(tmp_mean[:, 0]))], ['rank 5', 'rank 20', 'rank 50'])
        plt.ylabel('Log likelihood')
        plt.title(f"num {n}")
        plt.legend()

        # Show graphic
        plt.show()





