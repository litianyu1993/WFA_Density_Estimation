import numpy as np
import torch
from experiment_UCR import  labelize, sliding_window
import pickle
import os
from flows import FlowDensityEstimator

if __name__ == '__main__':
    L = 60
    exp_folder = './UCRArchive_2018/Adiac/'

    test = np.genfromtxt(exp_folder + 'Adiac_TEST.tsv', delimiter='\t')
    train = np.genfromtxt(exp_folder + 'Adiac_TRAIN.tsv', delimiter='\t')

    train = sliding_window(train, window_size=L)
    test = sliding_window(test, window_size=L)
    test = labelize(test, tensorize=True)
    train = labelize(train, tensorize=True)


    baseline = 'realnvp'

    class_idx_list = [22, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 17, 23, 25, 26, 29, 34, 36]
    # try_class_idx = [22]

    # priors = {}
    # total_num_examples = 0.
    # for class_key in class_idx_list:
    #     total_num_examples += len(train[class_key])
    #     priors[class_key] = len(train[class_key])
    # for class_key in class_idx_list:
    #     priors[class_key] = priors[class_key]/total_num_examples

    if baseline is not None:
        for class_key in class_idx_list:
            print(train[class_key].shape)
            flow = FlowDensityEstimator(baseline, num_inputs=train[class_key].shape[-1], num_hidden=64, num_blocks=5, num_cond_inputs=None, act='relu', device='cpu')
            tmp_train =  train[class_key].squeeze()
            tmp_test = test[class_key].squeeze()
            train_lik, test_lik = flow.train({'train': train[class_key], 'test': test[class_key]}, batch_size=train[class_key].shape[0], epochs=50)

            # train_lik = train_lik * priors[class_key]
            # test_lik = test_lik * priors[class_key]
            print('[%d] Train: %f Test: %f'%(class_key, -train_lik, -test_lik))
    exit()
    test_likelihood = {}
    for class_key in class_idx_list:
        for model_key in class_idx_list:
            outfile = open(os.path.join('results','UCR_Adiac','density_wfa_finetune',str(model_key)), 'rb')
            dwfa_finetune = pickle.load(outfile)
            outfile.close()
            likelihood = dwfa_finetune.eval_likelihood(test[class_key])
            if class_key not in test_likelihood:
                test_likelihood[class_key] = {}
            test_likelihood[class_key][model_key] = likelihood


    errors = 0.
    totals = 0.
    for ind, class_key in enumerate(class_idx_list):
        first = True
        for model_key in class_idx_list:
            if first:
                likelihood = test_likelihood[class_key][model_key].reshape(-1, 1)
                first = False
            else:
                likelihood = torch.cat([likelihood, test_likelihood[class_key][model_key].reshape(-1, 1)], dim = 1)
        print(likelihood.shape)
        # likelihood = torch.tensor(likelihood)
        pred_class = torch.argmax(likelihood, dim=1)
        for pred in pred_class:
            if pred != ind: errors += 1
            totals += 1
        print(pred_class)
    print('Error rate: ', errors/totals)


