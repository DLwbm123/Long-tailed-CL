import os
import sys


def add_gvalign_src(gvalign_root):
    src = os.path.join(os.path.abspath(gvalign_root), "src")
    if not os.path.isdir(src):
        raise FileNotFoundError("GVAlign src directory not found: {}".format(src))
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def load_gvalign_protocol(gvalign_root, data_root, dataset, num_tasks, nc_first_task, batch_size, num_workers):
    os.environ["GVALIGN_DATA_ROOT"] = os.path.abspath(data_root)
    add_gvalign_src(gvalign_root)
    from datasets.data_loader import get_loaders

    return get_loaders(
        [dataset],
        num_tasks=num_tasks,
        nc_first_task=nc_first_task,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )


def make_gvalign_exemplars(transform, class_indices, num_exemplars_per_class, exemplar_selection):
    from datasets.exemplars_dataset import ExemplarsDataset

    return ExemplarsDataset(
        transform,
        class_indices,
        num_exemplars_per_class=num_exemplars_per_class,
        exemplar_selection=exemplar_selection,
    )


def build_gvalign_resnet32_backbone(gvalign_root):
    add_gvalign_src(gvalign_root)
    from networks.resnet32 import resnet32
    import torch.nn as nn

    backbone = resnet32()
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Sequential()
    return backbone, feature_dim
