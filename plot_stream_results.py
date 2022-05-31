import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 7}
if __name__ == '__main__':
    methods = ['wfa', 'lstm', 'gru']
    datas = ['poker', 'rialto', 'weather', 'Elec2', 'sea', 'mixedDrift', 'hyperplane', 'chess']
    fig, axs = plt.subplots(2, 4)
    for i, data in enumerate(datas):
        for j, method in enumerate(methods):
            # try:
            if data =='weather' or data =='Elec2' or data == 'poker' or data =='covType' or data =='rialto':
                file_dir = os.path.dirname(os.path.realpath('__file__'))
                file_dir = os.path.join(file_dir, 'realWorld', data)
            else:
                file_dir = os.path.dirname(os.path.realpath('__file__'))
                file_dir = os.path.join(file_dir, 'artificial', data)

            with open(os.path.join((file_dir), f'{method}_results'), 'rb') as f:
                results = pickle.load(f)

            # if data == 'mixedDrift' and method == 'gru':
            #     print(results)
            # print(results['args'])
            results = np.asarray(results['results'])
            if data == 'poker': ls = np.arange(len(results))*1000
            elif data == 'chess' and method == 'wfa': ls = np.arange(len(results))*500
            else:  ls = np.arange(len(results))*100
            # print(i, j, results.shape)
            if i > 3:
                col = i - 4
                row = 1
            else:
                col = i
                row = 0

            axs[row, col].plot(ls, results[:, 2], label=f'RNADE-{method}')
            axs[row, col].title.set_text(data)
            if row == 1:
                axs[row, col].set_xlabel('Sequence Size')
            if col == 0:
                axs[row, col].set_ylabel('AUC')
            # plt.plot(results[:, 2], label = method)
            print(method, data, results[-1, 2], results[-1, -1])
            # except:
            #     print(method, data, 'not found')
    plt.legend()
    # plt.title(f'{data}_AUC')
    plt.show()