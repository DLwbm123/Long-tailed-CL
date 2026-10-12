"""Declared nonofficial DAT/Hao implementations; no data access or analytic-head substitution."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

VERSION = 'dat_hao_paper_medical_v1'
METHODS = ('DAT_paper', 'Hao_paper')


def grow_head(old, old_classes, classes, dim):
    head = nn.Linear(dim, len(classes)).to(next(old.parameters()).device if old is not None else 'cpu')
    if old is not None:
        with torch.no_grad():
            for i, c in enumerate(old_classes):
                j = classes.index(c);head.weight[j].copy_(old.weight[i]);head.bias[j].copy_(old.bias[i])
    return head


class HyperAdapter(nn.Module):
    """Layer identity modulates shared generators; declared completion of unspecified H(i)."""
    def __init__(self, dim=768, bottleneck=64, depth=12):
        super().__init__()
        self.down = nn.Parameter(torch.empty(bottleneck, dim))
        self.up = nn.Parameter(torch.zeros(dim, bottleneck))
        self.layer_embedding = nn.Parameter(torch.ones(depth, 2, bottleneck))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
    def forward(self, x, layer):
        d, u = self.layer_embedding[layer]
        return F.linear(F.relu(F.linear(x, self.down * d[:, None])), self.up * u[None, :])


def adapted_features(core, hyper, x):
    x = core.patch_embed(x);x = core._pos_embed(x);x = core.patch_drop(x);x = core.norm_pre(x)
    for i, block in enumerate(core.blocks):
        x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
        z = block.norm2(x)
        x = x + block.drop_path2(block.ls2(block.mlp(z))) + hyper(z, i)
    return core.forward_head(core.norm(x), pre_logits=True)


class DatEntry(nn.Module):
    def __init__(self, classes, dim=768, bottleneck=64, depth=12):
        super().__init__();self.classes = list(classes)
        self.hyper = HyperAdapter(dim, bottleneck, depth)
        self.imbalanced = nn.Linear(dim, len(classes));self.balanced = nn.Linear(dim, len(classes))
    def expand(self, classes):
        union = self.classes + [c for c in classes if c not in self.classes]
        self.imbalanced = grow_head(self.imbalanced, self.classes, union, self.imbalanced.in_features)
        self.balanced = grow_head(self.balanced, self.classes, union, self.balanced.in_features)
        self.classes = union


def kmeans_labels(features, k, seed):
    # Declared Lloyd/kmeans++ completion; no shared environment changes.
    x = features.detach().float().cpu();best=None;best_inertia=float('inf')
    if not torch.isfinite(x).all() or k<1 or len(x)<k:raise ValueError('Kmeans finite support')
    for restart in range(10):
        generator=torch.Generator().manual_seed(seed+restart*1009)
        selected=[int(torch.randint(len(x),(1,),generator=generator))]
        distance=(x-x[selected[0]]).square().sum(1)
        while len(selected)<k:
            probability=distance.clone();probability[selected]=0
            if probability.sum()>0:index=int(torch.multinomial(probability,1,generator=generator))
            else:index=next(i for i in range(len(x)) if i not in selected)
            selected.append(index);distance=torch.minimum(distance,(x-x[index]).square().sum(1))
        centres=x[selected].clone()
        for iteration in range(100):
            distances=torch.cdist(x,centres).square();labels=distances.argmin(1)
            changed=centres.clone();nearest=distances.min(1).values
            for i in range(k):
                rows=x[labels==i]
                if len(rows):changed[i]=rows.mean(0)
                else:
                    index=int(nearest.argmax());changed[i]=x[index];nearest[index]=-1
            movement=float((changed-centres).square().sum());centres=changed
            if movement<=1e-8:break
        distances=torch.cdist(x,centres).square();labels=distances.argmin(1);inertia=float(distances.min(1).values.sum())
        if inertia<best_inertia:best_inertia=inertia;best=labels
    return best


def cluster_accuracy(features, targets, seed):
    classes, y = torch.unique(targets.cpu(), sorted=True, return_inverse=True)
    if len(classes) < 2 or len(targets) < len(classes):raise ValueError('Current-task clustering support')
    z = kmeans_labels(features, len(classes), seed).numpy()
    matrix = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for a, b in zip(z, y.tolist()):matrix[a, b] += 1
    a, b = linear_sum_assignment(-matrix.numpy())
    return float(matrix[a, b].sum()) / len(y)


def balanced_indices(targets, batches, batch_size, seed):
    """All current classes appear equally in each balanced batch; nominal 24 is a cap."""
    generator = torch.Generator().manual_seed(seed);classes = torch.unique(targets, sorted=True)
    count = batch_size // len(classes)
    if count < 1:raise ValueError('Balanced batch smaller than current label count')
    groups = [torch.where(targets == c)[0] for c in classes]
    for _ in range(batches):
        result = torch.cat([g[torch.randint(len(g), (count,), generator=generator)] for g in groups])
        yield result[torch.randperm(len(result), generator=generator)].tolist()


def dat_loss(core, entry, key, x, y, xb, yb, counts, temperature=2., beta=.9999, teacher=None):
    device = x.device;classes = torch.tensor(entry.classes, device=device)
    local = (y[:, None] == classes[None, :]).long().argmax(1)
    local_b = (yb[:, None] == classes[None, :]).long().argmax(1)
    if not (y[:, None] == classes[None, :]).any(1).all() or not (yb[:, None] == classes[None, :]).any(1).all():
        raise ValueError('DAT classifier label map')
    with torch.no_grad():base = core(x);base_b = core(xb)
    z = entry.imbalanced(adapted_features(core, entry.hyper, x))
    zb = entry.balanced(base_b);q = F.softmax(entry.balanced(base).detach() / temperature, dim=1)
    weights = (1-beta) / (-torch.expm1(counts.to(device=device,dtype=z.dtype) * math.log(beta)))
    lbd = temperature**2 * (weights[None,:] * q * (q.clamp_min(1e-30).log() - F.log_softmax(z/temperature, dim=1))).sum(1).mean()
    pieces = dict(imbalanced_ce=F.cross_entropy(z,local), balanced_ce=F.cross_entropy(zb,local_b),
                  balanced_distillation=lbd, key_match=-F.cosine_similarity(base,key[None,:]).mean())
    if teacher is not None:
        with torch.no_grad():old = teacher.imbalanced(adapted_features(core,teacher.hyper,x))
        positions = [entry.classes.index(c) for c in teacher.classes]
        pieces['task_distillation'] = F.mse_loss(z[:,positions],old)
    return sum(pieces.values()), pieces


def dat_predict(core, entries, keys, key_entries, x, seen):
    with torch.no_grad():base = core(x)
    chosen = F.normalize(base,dim=1) @ F.normalize(keys,dim=1).T
    entry_ids = key_entries[chosen.argmax(1)]
    scores = x.new_full((len(x),seen), -1e9)
    for i in entry_ids.unique().tolist():
        use = torch.where(entry_ids == i)[0];entry = entries[i]
        logits = entry.imbalanced(adapted_features(core,entry.hyper,x[use]))
        scores[use[:,None],torch.tensor(entry.classes,device=x.device)[None,:]] = logits
    return scores, entry_ids


class CosineHead(nn.Module):
    def __init__(self, classes, dim=512):
        super().__init__();self.weight = nn.Parameter(torch.empty(classes,dim));self.scale = nn.Parameter(torch.tensor(10.))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    def forward(self, features):
        return self.scale * F.linear(F.normalize(features,dim=1),F.normalize(self.weight,dim=1))
    def expand(self, classes):
        if classes <= len(self.weight):raise ValueError('Classifier must grow')
        weight = self.weight.new_empty(classes,self.weight.shape[1]);nn.init.kaiming_uniform_(weight,a=math.sqrt(5))
        weight[:len(self.weight)] = self.weight.detach();self.weight = nn.Parameter(weight)


class HaoNet(nn.Module):
    def __init__(self, core, classes):
        super().__init__();self.core = core;self.head = CosineHead(classes)
    def forward(self,x):
        features = self.core(x);return self.head(features),features


def hao_loss(model, x, targets, counts, old, inherent_counts, teacher=None, alpha=.5, lambda_d=.4, lambda_k=.75, margin=.4):
    logits, features = model(x)
    if len(counts) != logits.shape[1] or (counts <= 0).any():raise ValueError('Actual replay/current support required')
    adjustment = counts.to(logits).log()
    if old:adjustment[:old] += math.log(alpha)
    cbc = F.cross_entropy(logits + adjustment,targets)
    dm = logits.new_zeros(());kd = logits.new_zeros(())
    if old:
        if teacher is None:raise ValueError('Missing frozen old teacher')
        with torch.no_grad():previous,_ = teacher(x)
        kd = (logits[:,:old] - previous).abs().sum(1).mean()
        use = targets < old
        if use.any():
            weights = F.normalize(model.head.weight,dim=1)
            radius = inherent_counts.to(logits);radius = radius/radius.sum()
            ranges = F.normalize(weights + torch.randn_like(weights)*radius[:,None],dim=1)
            anchors = F.normalize(features[use],dim=1);labels = targets[use]
            same = (anchors*weights[labels]).sum(1)
            inter = F.relu(anchors@ranges[old:].T - same[:,None] + margin).sum()
            intra = F.relu((ranges[labels]*weights[labels]).sum(1)-same).sum()
            dm = inter+intra
    return cbc+lambda_d*dm+lambda_k*kd,dict(cil_balanced_ce=cbc,distribution_margin=dm,logit_l1_distillation=kd)


def herding_indices(features, limit):
    features = F.normalize(features.detach().float().cpu(),dim=1)
    limit = min(limit,len(features));target = features.mean(0);total = torch.zeros_like(target)
    selected=[];available=torch.ones(len(features),dtype=torch.bool)
    for step in range(limit):
        distance = ((total[None,:]+features)/(step+1)-target[None,:]).square().sum(1)
        distance[~available] = float('inf');index = int(distance.argmin())
        selected.append(index);available[index] = False;total += features[index]
    return selected
