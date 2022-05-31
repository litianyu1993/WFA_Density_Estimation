import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import optim
import numpy as np
# from utils import negative_log_likelihood

def train(model, device, train_loader, optimizer, X, loss_function = F.mse_loss, rescale = False, transformer = False):
    error = 0.
    counts = 0
    for batch_idx, x in enumerate(train_loader):
        if rescale:
            model.update_scale()
        if transformer:
            x = torch.transpose(x, 0, 1)
        x = x.to(device)
        optimizer.zero_grad()

        # loss = model.negative_log_likelihood(model, x)
        loss = model.lossfunc(x)

        loss.backward()
        # for param in model.parameters():
        #
        #     print(param.grad)
        #     # param.grad = torch.nan_to_num(param.grad)
        if hasattr(model, 'init_w'):
            model.init_w.grad = torch.nan_to_num(model.init_w.grad)
            model.A.grad = torch.nan_to_num(model.A.grad)
            model.mu_out.weight.grad =  torch.nan_to_num(model.mu_out.weight.grad)
            # print(model.mu_out.weight.grad[0])
            # print(model.A.grad[0])
            model.mu_out.bias.grad = torch.nan_to_num(model.mu_out.bias.grad)
            model.sig_out.weight.grad = torch.nan_to_num(model.sig_out.weight.grad)
            model.sig_out.bias.grad = torch.nan_to_num(model.sig_out.bias.grad)
            model.alpha_out.weight.grad = torch.nan_to_num(model.alpha_out.weight.grad)
            model.alpha_out.bias.grad = torch.nan_to_num(model.alpha_out.bias.grad)
            model.batchnrom.weight.grad = torch.nan_to_num(model.batchnrom.weight.grad)
            model.batchnrom.bias.grad = torch.nan_to_num(model.batchnrom.bias.grad)

        # print(model.init_w)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        # print(batch_idx, loss)
        # print(batch_idx, model.core_list[0].grad)
        # print(batch_idx, model.alpha_out.weight.grad)
        # print(model[0].weight.grad)
        optimizer.step()
        # total_norm = 0.
        # param_norm = model.A.grad.detach().norm(2)
        # total_norm += param_norm** 2
        # total_norm = total_norm ** (1. / 2)
        # print(total_norm)


        # print(batch_idx, model.negative_log_likelihood(x))
        error+= loss.detach().cpu().numpy()
        counts +=1
    error = error/counts
    return error

def validate(model, device, validation_loader, X, loss_function = F.mse_loss, transformer = False):
    error = 0.
    counts = 0
    with torch.no_grad():
        for x in validation_loader:
            if transformer:
                x = torch.transpose(x, 0, 1)
            x = x.to(device)
            # test_loss = negative_log_likelihood(model, x)
            test_loss = model.lossfunc(x)
            error += test_loss.detach().cpu().numpy()
            counts += 1

    error = error / counts
    return error


