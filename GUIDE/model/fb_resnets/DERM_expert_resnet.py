
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Expert_ResNet import ResNet as Expert_ResNet, Bottleneck


class AttentionBlock(nn.Module):
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


class DERM_ResNet(Expert_ResNet):
    def __init__(self, block, layers, num_experts, **kwargs):
        super().__init__(block, layers, num_experts, **kwargs)

        self.shallow_path_layer4s = self.layer4s
        layer3_output_dim = 256
        if 'layer3_output_dim' in kwargs and kwargs['layer3_output_dim'] is not None:
            layer3_output_dim = kwargs['layer3_output_dim']

        in_planes_for_layer4 = layer3_output_dim * block.expansion

        layer4_output_dim = 512
        if 'layer4_output_dim' in kwargs and kwargs['layer4_output_dim'] is not None:
            layer4_output_dim = kwargs['layer4_output_dim']

        deep_paths = []
        for _ in range(num_experts):
            self.inplanes = in_planes_for_layer4

            deeper_layer4 = nn.Sequential(
                self._make_layer(block, layer4_output_dim, layers[3], stride=2),
                AttentionBlock(layer4_output_dim * block.expansion)
            )
            deep_paths.append(deeper_layer4)

        self.deep_path_layer4s = nn.ModuleList(deep_paths)

        del self.layer4s

    def _separate_part(self, x, ind, gate):

        g = gate.view(-1, 1, 1, 1)

        shallow_out_feat = self.shallow_path_layer4s[ind](x)
        deep_out_feat = self.deep_path_layer4s[ind](x)

        out = (1 - g) * shallow_out_feat + g * deep_out_feat

        self.feat_before_GAP.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        if self.use_dropout:
            out = self.dropout(out)

        self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

    def forward(self, x, gates_per_expert=None, **kwargs):
        """
        This method is overridden to accept and pass the `gates_per_expert` argument.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        if self.share_layer3:
            out = self.layer3(out)

        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []

        for ind in range(self.num_experts):
            batch_gates = gates_per_expert[ind] if gates_per_expert is not None else torch.full((x.size(0), 1), 0.5,
                                                                                                device=x.device)

            input_for_part = out if self.share_layer3 else self.layer3s[ind](out)

            outs.append(self._separate_part(input_for_part, ind, batch_gates))

        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out,
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out