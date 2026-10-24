"""Frozen base-only optimizer and exact full-logical-batch output-gradient replay."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import ProjectionHeads, native_descriptor, paired_objective
from coupled import backward_coupled

ARMS = ['NATIVE_RIDGE', 'BASE_SUPERVISED', 'GLOBAL_REDUNDANCY', 'LOCAL_GLOBAL_REDUNDANCY']


class Learner(nn.Module):
    def __init__(self, upper, arm, means):
        super().__init__()
        self.upper, self.arm = upper, arm
        self.projectors = ProjectionHeads().to(next(upper.parameters()))
        self.head = nn.Parameter(F.normalize(means.to(next(upper.parameters())), dim=1).clone())
        self.projectors.requires_grad_(arm != 'BASE_SUPERVISED')
        self.head.requires_grad_(arm == 'BASE_SUPERVISED')
        self.eval()

    def objective(self, tokens, labels, receipt):
        n = len(labels)
        assert len(tokens) == 2*n
        if self.arm == 'BASE_SUPERVISED':
            logits = 25 * native_descriptor(tokens) @ F.normalize(self.head, dim=1).T
            return F.cross_entropy(logits, torch.cat((labels, labels)).to(logits.device))
        cells = [torch.tensor(receipt[k], device=tokens.device) for k in ('first_cells', 'second_cells')]
        return paired_objective(self.projectors, tokens[:n], tokens[n:], *cells,
                                local=self.arm == 'LOCAL_GLOBAL_REDUNDANCY')


def optimizer(model, cfg):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'], eps=cfg['adam_epsilon'],
                            betas=(.9, .999), foreach=False, fused=False)


def canonical_outputs(model, inputs, micro=8):
    reference = next(model.parameters())
    with torch.no_grad():
        return torch.cat([model.upper(inputs[i:i+micro].to(reference)) for i in range(0, len(inputs), micro)])


def update(model, opt, inputs, labels, receipt, cfg, count, record, audit=False):
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();tick = time.perf_counter()
    def logged(value):
        if value['check'] == 'coupled_completed_rows':
            for key, field in [('first_forward','first_forward_rows'),('replay_forward','replay_forward_rows'),('replay_backward','backward_rows')]:
                count[field] += value[key]
        record(value)
    loss = backward_coupled(model.upper, inputs,
                            lambda out: model.objective(out, labels, receipt), logged, cfg['microbatch'])
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
    record(dict(check='finite_clipped_gradient', norm=float(norm), PASS=bool(torch.isfinite(norm))))
    assert bool(torch.isfinite(norm))
    expected = [];audit_seconds=0.
    if audit:
        audit_tick=time.perf_counter()
        for group in opt.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:continue
                state = opt.state[parameter];step = int(state.get('step', 0)) + 1
                w = parameter.detach().cpu().numpy().copy();g = parameter.grad.detach().cpu().numpy()
                m = state['exp_avg'].detach().cpu().numpy().copy() if state else np.zeros_like(w)
                v = state['exp_avg_sq'].detach().cpu().numpy().copy() if state else np.zeros_like(w)
                m = .9*m + .1*g;v = .999*v + .001*g*g
                e = w*(1-group['lr']*group['weight_decay']) - group['lr']*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+group['eps'])
                expected.append((parameter, e))
        audit_seconds+=time.perf_counter()-audit_tick
    opt.step();count['updates'] += 1
    if audit:
        torch.cuda.synchronize();audit_tick=time.perf_counter()
        error = max(float(abs(p.detach().cpu().numpy()-e).max()) for p,e in expected)
        scaled = max(float((abs(p.detach().cpu().numpy()-e)/(cfg['operator_absolute_tolerance']+cfg['operator_relative_tolerance']*abs(e))).max()) for p,e in expected)
        record(dict(check='actual_same_gradient_AdamW', max_absolute_error=error, normalized_error=scaled, PASS=scaled<=1))
        assert scaled <= 1
        audit_seconds+=time.perf_counter()-audit_tick
    torch.cuda.synchronize()
    return dict(objective=loss, gradient_norm=float(norm), seconds=time.perf_counter()-tick-audit_seconds, qualification_audit_seconds=audit_seconds)


def counter():return dict(updates=0, first_forward_rows=0, replay_forward_rows=0, backward_rows=0)
