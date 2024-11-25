import torch
import torch.nn as nn
torch.manual_seed(0)


def define_G(yaml_conf, input_dim, output_dim, device):
    if "input_dim" in yaml_conf:
        input_dim = yaml_conf["input_dim"]
    if "output_dim" in yaml_conf:
        output_dim = yaml_conf["output_dim"]
        
    if yaml_conf["model"] == "bn_mlp_sigmoid":
        return BnMLP_sigmoid(yaml_conf, input_dim, output_dim).to(device)
    else:
        raise NotImplementedError(
            'Wrong model name %s (choose one from [lsr, softmax])' % yaml_conf["model"])


class BnMLP_sigmoid(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim*4, out_features=256, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x: batch_size x feat_dim
        x = torch.cat([x, x**2, torch.sin(x), torch.cos(x)], dim=-1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.relu(self.bn4(self.linear4(x)))
        logits = self.linear_pred(x)
        logits = self.sigmoid(logits)

        return logits

    def get_last_shared_layer(self):
        return self.linear_pred