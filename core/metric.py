import torch
import torch.nn as nn


class RegressionAccuracy(nn.Module):

    def __init__(self, choice='r2'):
        super(RegressionAccuracy, self).__init__()
        self.choice = choice

    def __call__(self, y_pred, y_true):
        y_true = y_true.float()
        if self.choice == 'r2':
            var_y = torch.var(y_true, unbiased=False)
            acc = 1.0 - torch.mean((y_pred - y_true)**2) / (var_y + 1e-9)
            return acc

        else:
            raise NotImplementedError(
                'Unknown loss function type %s ' % self.choice)
