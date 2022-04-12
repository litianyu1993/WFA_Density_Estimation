import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import optim
import numpy as np
# from utils import negative_log_likelihood

def train(model, device, train_loader, optimizer, X, loss_function = F.mse_loss, rescale = False):
    error = 0.
    counts = 0
    for batch_idx, x in enumerate(train_loader):
        if rescale:
            model.update_scale()
        x = x.to(device)
        optimizer.zero_grad()

        # loss = model.negative_log_likelihood(model, x)
        loss = model.lossfunc(x)

        loss.backward()
        # print(loss.grad)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
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

def validate(model, device, validation_loader, X, loss_function = F.mse_loss):
    error = 0.
    counts = 0
    with torch.no_grad():
        for x in validation_loader:
            x = x.to(device)
            # test_loss = negative_log_likelihood(model, x)
            test_loss = model.lossfunc(x)
            error += test_loss.detach().cpu().numpy()
            counts += 1

    error = error / counts
    return error


