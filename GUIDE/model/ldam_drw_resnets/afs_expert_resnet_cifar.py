# ==============================================================================
# BEGIN FILE: model/ldam_drw_resnets/afs_expert_resnet_cifar.py (FINAL VERSION)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert_resnet_cifar import ResNet_s, BasicBlock


class AttentionBlock(nn.Module):
    """一个简单的SE-Block实现，用于我们的精深通路。"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AFS_ResNet_s(ResNet_s):
    """
    我们的核心模型：自适应焦点切换ResNet专家网络。
    它继承自 expert_resnet_cifar.py 中的 ResNet_s。
    """

    def __init__(self, block, num_blocks, num_experts, **kwargs):
        super().__init__(block, num_blocks, num_experts, **kwargs)

        self.shallow_path_layer3s = self.layer3s

        if kwargs.get('reduce_dimension', False):
            layer2_output_dim = kwargs.get('layer2_output_dim', 24)
        else:
            layer2_output_dim = kwargs.get('layer2_output_dim', 32)

        in_planes_for_layer3 = layer2_output_dim * block.expansion

        if kwargs.get('reduce_dimension', False):
            layer3_output_dim = kwargs.get('layer3_output_dim', 48)
        else:
            layer3_output_dim = kwargs.get('layer3_output_dim', 64)

        deep_paths = []
        for _ in range(num_experts):
            self.in_planes = in_planes_for_layer3
            deeper_layer3 = nn.Sequential(
                self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2),
                # 额外增加一个 BasicBlock
                block(self.next_in_planes, layer3_output_dim),
                AttentionBlock(layer3_output_dim * block.expansion)
            )
            deep_paths.append(deeper_layer3)

            # deep_path = nn.Sequential(
            #     self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2),
            #     AttentionBlock(layer3_output_dim * block.expansion)
            # )
            # deep_paths.append(deep_path)
        self.deep_path_layer3s = nn.ModuleList(deep_paths)

        self.in_planes = self.next_in_planes

        del self.layer3s

    def _separate_part(self, x, ind, gate):
        out = x
        out = (self.layer2s[ind])(out)
        g = gate.view(-1, 1, 1, 1)
        shallow_out = self.shallow_path_layer3s[ind](out)
        deep_out = self.deep_path_layer3s[ind](out)
        out = (1 - g) * shallow_out + g * deep_out

        self.feat_before_GAP.append(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

    def forward(self, x, gates_per_expert=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []

        for ind in range(self.num_experts):
            batch_gates = gates_per_expert[ind] if gates_per_expert is not None else torch.full((x.size(0), 1), 0.5,
                                                                                                device=x.device)
            outs.append(self._separate_part(out, ind, batch_gates))

        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {"output": final_out, "feat": self.feat, "logits": torch.stack(outs, dim=1)}
        else:
            return final_out


def afs_resnet32(**kwargs):
    return AFS_ResNet_s(BasicBlock, [5, 5, 5], **kwargs)