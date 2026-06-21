# ==============================================================================
# BEGIN FILE: model/model.py (FINAL VERSION)
# ==============================================================================
import torch
import torch.nn as nn
from base import BaseModel
from .fb_resnets import ResNet, ResNeXt, Expert_ResNet, Expert_ResNeXt
from .ldam_drw_resnets import resnet_cifar, expert_resnet_cifar

# 【新增】导入我们的新模型组件
from .ldam_drw_resnets.afs_expert_resnet_cifar import AFS_ResNet_s
from .strategy_controller import StrategyController
from .fb_resnets.DERM_expert_resnet import DERM_ResNet # <-- 新增导入
from .fb_resnets.Expert_ResNet import Bottleneck # <-- 为了保险起见，可以显式导入
from .fb_resnets.DERM_expert_resnet import DERM_ResNet # <-- 新增导入
from .fb_resnets.DERM_expert_resnet import DERM_ResNet
from .fb_resnets.Expert_ResNet import Bottleneck


class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None:
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, **kwargs):

        return self.backbone(x, **kwargs)

class DERM_ResNet50Model(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes, None)
        self.num_experts = kwargs.get('num_experts', 1)
        self.backbone = DERM_ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
        self.controllers = nn.ModuleList([StrategyController(num_classes=num_classes) for _ in range(self.num_experts)])

class DERM_ResNet152Model(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes, None)
        self.num_experts = kwargs.get('num_experts', 1)
        self.backbone = DERM_ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
        self.controllers = nn.ModuleList([StrategyController(num_classes=num_classes) for _ in range(self.num_experts)])


class ResNet10Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes,
                                          use_norm=use_norm, reduce_dimension=reduce_dimension,
                                          layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim,
                                          **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes,
                                                 use_norm=use_norm, reduce_dimension=reduce_dimension,
                                                 layer3_output_dim=layer3_output_dim,
                                                 layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)


class ResNet32Model(Model):  # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes,
                                                  reduce_dimension=reduce_dimension,
                                                  layer2_output_dim=layer2_output_dim,
                                                  layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5],
                                                         num_classes=num_classes, reduce_dimension=reduce_dimension,
                                                         layer2_output_dim=layer2_output_dim,
                                                         layer3_output_dim=layer3_output_dim, use_norm=use_norm,
                                                         num_experts=num_experts, **kwargs)


class AFS_ResNet32Model(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes, None)
        self.num_experts = kwargs.get('num_experts', 1)
        self.backbone = AFS_ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, **kwargs)
        self.controllers = nn.ModuleList([StrategyController(num_classes=num_classes) for _ in range(self.num_experts)])


class ResNet32Model_B(Model):  # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlockB, [5, 5, 5], num_classes=num_classes,
                                                  reduce_dimension=reduce_dimension,
                                                  layer2_output_dim=layer2_output_dim,
                                                  layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlockB, [5, 5, 5],
                                                         num_classes=num_classes, reduce_dimension=reduce_dimension,
                                                         layer2_output_dim=layer2_output_dim,
                                                         layer3_output_dim=layer3_output_dim, use_norm=use_norm,
                                                         num_experts=num_experts, **kwargs)


class ResNet34Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [3, 4, 6, 3], dropout=None, num_classes=num_classes,
                                          use_norm=use_norm, reduce_dimension=reduce_dimension,
                                          reduce_first_kernel=True,
                                          layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim,
                                          **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.BasicBlock, [3, 4, 6, 3], dropout=None,
                                                 num_classes=num_classes,
                                                 use_norm=use_norm, reduce_dimension=reduce_dimension,
                                                 reduce_first_kernel=True,
                                                 layer3_output_dim=layer3_output_dim,
                                                 layer4_output_dim=layer4_output_dim,
                                                 num_experts=num_experts, **kwargs)


class ResNet50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes,
                                          reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim,
                                          layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None,
                                                 num_classes=num_classes, reduce_dimension=reduce_dimension,
                                                 layer3_output_dim=layer3_output_dim,
                                                 layer4_output_dim=layer4_output_dim, use_norm=use_norm,
                                                 num_experts=num_experts, **kwargs)


class ResNeXt50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4,
                                            dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension,
                                            layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim,
                                            use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32,
                                                   width_per_group=4, dropout=None, num_classes=num_classes,
                                                   reduce_dimension=reduce_dimension,
                                                   layer3_output_dim=layer3_output_dim,
                                                   layer4_output_dim=layer4_output_dim, num_experts=num_experts,
                                                   use_norm=use_norm, **kwargs)


class ResNet101Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes,
                                          reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim,
                                          layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 23, 3], dropout=None,
                                                 num_classes=num_classes, reduce_dimension=reduce_dimension,
                                                 layer3_output_dim=layer3_output_dim,
                                                 layer4_output_dim=layer4_output_dim, use_norm=use_norm,
                                                 num_experts=num_experts, **kwargs)


class ResNet152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 share_layer3=False, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes,
                                          reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim,
                                          layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 8, 36, 3], dropout=None,
                                                 num_classes=num_classes, reduce_dimension=reduce_dimension,
                                                 layer3_output_dim=layer3_output_dim,
                                                 layer4_output_dim=layer4_output_dim, share_layer3=share_layer3,
                                                 use_norm=use_norm, num_experts=num_experts, **kwargs)


class ResNeXt152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4,
                                            dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension,
                                            layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32,
                                                   width_per_group=4, dropout=None, num_classes=num_classes,
                                                   reduce_dimension=reduce_dimension,
                                                   layer3_output_dim=layer3_output_dim,
                                                   layer4_output_dim=layer4_output_dim, num_experts=num_experts)