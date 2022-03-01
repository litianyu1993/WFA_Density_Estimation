import numpy as np
import torch
from experiment_UCR import  labelize
import pickle

if __name__ == '__main__':
    exp_folder = './UCRArchive_2018/Adiac/'

    test = np.genfromtxt(exp_folder + 'Adiac_TEST.tsv', delimiter='\t')
    test = labelize(test, tensorize=True)

    class_idx_list = [2, 15]

    test_likelihood = {}
    for class_key in class_idx_list:
        for model_key in class_idx_list:
            outfile = open(exp_folder + 'density_wfa_finetune' + str(model_key), 'rb')
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



