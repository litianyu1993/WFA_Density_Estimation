import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import optim
import numpy as np

def train(model, device, train_loader, optimizer, X, loss_function = F.mse_loss, rescale = False):
    error = []
    for batch_idx, x in enumerate(train_loader):
        if rescale:
            model.update_scale()

        x = x.to(device)
        optimizer.zero_grad()


        loss = model.negative_log_likelihood(x)

        loss.backward()
        # print(loss.grad)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        # print(batch_idx, loss)
        # print(batch_idx, model.core_list[0].grad)
        # print(batch_idx, model.alpha_out.weight.grad)
        # print(model[0].weight.grad)
        optimizer.step()


        # print(batch_idx, model.negative_log_likelihood(x))
        error.append(loss)
    return torch.mean(model.eval_likelihood(X))

def validate(model, device, validation_loader, X, loss_function = F.mse_loss):
    all_losses = []
    with torch.no_grad():
        for x in validation_loader:
            x = x.to(device)
            test_loss = model.negative_log_likelihood(x)
            all_losses.append(test_loss)
    return  torch.mean(model.eval_likelihood(X))


