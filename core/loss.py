import torch
import torch.nn as nn


class RegressionLoss(nn.Module):

    def __init__(self, choice='mse'):
        super(RegressionLoss, self).__init__()
        self.choice = choice

    def __call__(self, y_pred, y_true):

        if self.choice == 'mse':
            loss = torch.mean((y_pred - y_true)**2)
            return loss

        elif self.choice == 'mae':
            loss = torch.mean(torch.abs(y_pred - y_true))
            return loss

        else:
            raise NotImplementedError(
                'Unknown loss function type %s ' % self.choice)
