from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 7}
if __name__ == '__main__':

    with open('exp_drift_hmm.results_ranks_4', 'rb') as f:
        all_results = pickle.load(f)

    mean_ground = []
    mean_model = []
    std_ground = []
    std_model = []
    for r in [2, 20, 40, 80, 160, 320, 640, 1280]:
        ground_truth = []
        model_output = []
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
            for result  in all_results:
                if result['r'] == r and result['seed'] == seed:

                    ground_truth.append(savgol_filter(result['ground'], 51, 1))
                    tmp = result['ground'].reshape(-1) - result['model_output'].reshape(-1)
                    model_output.append(savgol_filter(np.abs(tmp), 11, 1))
                    # model_output.append(np.abs(tmp))
        # ground_truth_r = np.asarray(ground_truth).reshape(len(ground_truth), -1)

        model_output_r = np.asarray(model_output)

        mean_model.append(np.mean(model_output_r, axis = 0))
        # mean_ground.append(np.mean(ground_truth_r, axis = 0))
        # print(model_output_r.shape)
        std_model.append(np.std(model_output_r, axis=0))
        # std_ground.append(np.std(ground_truth_r, axis=0))


    # mean_ground = np.mean(np.asarray(mean_ground), axis = 0)
    # plt.plot(savgol_filter(mean_ground, 101, 3), label = 'ground truth')
    for i, r in enumerate([2, 20, 40, 80, 160, 320, 640, 1280]):
        # mean_model[i] = np.log(mean_model[i])
        # std_model[i] = np.log(std_model[i])
        if r == 20: plt.plot(mean_model[i], color = 'black', label = f'r={r}')
        else:
            plt.plot(mean_model[i], label = f'r={r}')
        # plt.plot(mean_model[i] - std_model[i], label = '1')
        # plt.plot(mean_model[i] + std_model[i], label = '2')
        # print(std_model[i])
        print(mean_model[i][-1])
        lower = mean_model[i] - std_model[i]
        # for j in range(len(lower)):
        #     lower[j] = np.min([lower[j], 1e-3])
        if r == 20:
            plt.fill_between(np.arange(len(model_output[i])), lower, mean_model[i] + std_model[i], color = 'black', alpha=0.2)
        else:
            plt.fill_between(np.arange(len(model_output[i])), lower, mean_model[i] + std_model[i],
                             alpha=0.2)
        # plt.plot(mean_model[i], label=f'r={r}')
    plt.ylim(0.01, 200)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Sequence Size')
    plt.ylabel('Log Likelihood')
    plt.show()

