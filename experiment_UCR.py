from Density_WFA_finetune import learn_density_WFA
import torch
import numpy as np
import pickle
import time

def sliding_window(X, window_size = 5):
    final_data = []
    for x in X:
        for j in range(1, len(x) - 1 - window_size):
            tmp = x[j:j+window_size]
            tmp = np.insert(tmp, 0, x[0])
            final_data.append(tmp)
    return np.asarray(final_data)


def labelize(X, tensorize = True):
    final_data = {}
    for x in X:
        if x[0] not in final_data:
            final_data[int(x[0])] = [x[1:]]
        else:
            final_data[int(x[0])].append(x[1:])

    for key in final_data.keys():
        if tensorize:
            final_data[key] = torch.tensor(np.asarray(final_data[key]))
            final_data[key] = final_data[key].reshape(final_data[key].shape[0], 1, final_data[key].shape[1])
        else:
            final_data[key] = np.asarray(final_data[key])
    return final_data


if __name__ == '__main__':
    load = False
    load_WFA = False
    data_label = [['train_l', 'test_l'], ['train_2l', 'test_2l'], ['train_2l1', 'test_2l1']]
    validation_split = 0.8
    l = 4  # length in trianning (l, 2l, 2l+1)
    Ls = [l, 2 * l, 2 * l + 1]


    model_params = {
        'd': 5,
        'xd': 1,
        'r': 20,
        'lr': 0.001,
        'epochs': 200,
        'fine_tune_epochs': 20,
        'fine_tune_lr':0.001,
        'batch_size': 256,
        'double_precision': True
    }

    model_params['mixture_n'] = model_params['r']

    exp_name = str(model_params)
    exp_folder = 'UCRArchive_2018/Adiac/'

    train = np.genfromtxt(exp_folder + 'Adiac_TRAIN.tsv', delimiter='\t')
    test = np.genfromtxt(exp_folder + 'Adiac_TEST.tsv', delimiter='\t')
    test = labelize(test, tensorize=True)

    train_x_tmp = sliding_window(train)
    train_x_tmp = labelize(train_x_tmp)


    for key in train_x_tmp:
        if not load:
            DATA = {}
            for k in range(len(Ls)):
                L = Ls[k]
                train_x_all = sliding_window(train, window_size=L)
                np.random.shuffle(train_x_all)
                train_x_all = labelize(train_x_all, tensorize=True)
                train_x_current_class = train_x_all[key]
                sep = int(validation_split * len(train_x_current_class))
                train_x = train_x_current_class[:sep]
                vali_x = train_x_current_class[sep:]
                # print(train_x.shape, vali_x.shape)
                DATA[data_label[k][0]] = train_x
                DATA[data_label[k][1]] = vali_x

            out_file_name = exp_folder + 'density_wfa'+str(key)
            dwfa_finetune = learn_density_WFA(DATA, model_params, l, out_file_name = out_file_name, load_WFA = load_WFA)


            outfile = open(exp_folder + 'density_wfa_finetune'+str(key), 'wb')
            pickle.dump(dwfa_finetune, outfile)
            outfile.close()
        else:
            outfile = open(exp_folder + 'density_wfa_finetune' + str(key), 'rb')
            dwfa_finetune = pickle.load(outfile)
            outfile.close()
        test_likelihood = {}
        # print(dwfa_finetune.get_transition_norm(), dwfa_finetune.scale)
        for i in range(dwfa_finetune.A.shape[1]):
            # u, s, v = torch.linalg.svd(A[:, i, :])
            s = torch.linalg.svdvals(dwfa_finetune.A[:, i, :])
        for key2 in train_x_tmp:
            # if key == 21:
            likelihood = dwfa_finetune.eval_likelihood(test[key2])
            test_likelihood[key2] = likelihood
            # print(likelihood)
            print(str(key2)+' class average log likelihood')
            print(torch.mean(likelihood))
            print(torch.std(likelihood))
        # outfile = open(exp_folder + 'density_wfa' + str(key)+'test_likelihood'+'model', 'wb')
        # pickle.dump(test_likelihood, outfile)
        # outfile.close()


            # DATA[data_label[k][0]] = train_x
            # DATA[data_label[k][1]] = test_x

