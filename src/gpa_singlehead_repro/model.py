import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCILModel(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.classifier = None
        self.task_cls = torch.LongTensor([])
        self.task_offset = torch.LongTensor([])

    def expand(self, task_sizes, init_new_weight=None, init_new_bias=None):
        num_seen = int(sum(task_sizes))
        old_classifier = self.classifier
        device = next(self.backbone.parameters()).device
        new_classifier = nn.Linear(self.feature_dim, num_seen).to(device)

        nn.init.kaiming_normal_(new_classifier.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(new_classifier.bias)

        old_dim = 0
        if old_classifier is not None:
            old_dim = old_classifier.out_features
            with torch.no_grad():
                new_classifier.weight[:old_dim].copy_(old_classifier.weight.data)
                new_classifier.bias[:old_dim].copy_(old_classifier.bias.data)

        if init_new_weight is not None:
            with torch.no_grad():
                new_classifier.weight[old_dim:num_seen].copy_(init_new_weight.to(device))
        if init_new_bias is not None:
            with torch.no_grad():
                new_classifier.bias[old_dim:num_seen].copy_(init_new_bias.to(device))

        self.classifier = new_classifier
        self.task_cls = torch.LongTensor(list(task_sizes)).to(device)
        self.task_offset = torch.cat(
            [torch.LongTensor(1).zero_().to(device), self.task_cls.cumsum(0)[:-1]]
        )
        return old_dim

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def get_copy(self):
        return copy.deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(copy.deepcopy(state_dict))


def normalize_rows(x, eps=1e-12):
    return F.normalize(x, p=2, dim=1, eps=eps)
