import numpy as np
import torch
from experiment_UCR import  labelize
import pickle

if __name__ == '__main__':
    exp_folder = './UCRArchive_2018/Beef/'

    test = np.genfromtxt(exp_folder + 'Beef_TEST.tsv', delimiter='\t')
    train = np.genfromtxt(exp_folder + 'Beef_TRAIN.tsv', delimiter='\t')
    test = labelize(test, tensorize=True)
    train = labelize(train, tensorize=True)



    # class_idx_list = [3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 17, 23, 25, 26, 29, 34, 36]
    class_idx_list = [1, 2, 3, 4]

    priors = {}
    total_num_examples = 0.
    for class_key in class_idx_list:
        total_num_examples += len(train[class_key])
        priors[class_key] = len(train[class_key])
    for class_key in class_idx_list:
        priors[class_key] = priors[class_key]/total_num_examples
    print(priors)

    test_likelihood = {}
    for class_key in class_idx_list:
        for model_key in class_idx_list:
            # print(model_key)
            outfile = open(exp_folder + 'density_wfa_finetune' + str(model_key), 'rb')
            dwfa_finetune = pickle.load(outfile)
            outfile.close()
            likelihood = torch.tensor(dwfa_finetune.eval_likelihood(test[class_key]) + np.log(priors[class_key]))
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



