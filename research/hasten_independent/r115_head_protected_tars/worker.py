"""Constrain actual TARS Adam proposals with matched head-feedback controls."""
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import write


def cone_project(displacement, gradient, margin=.1):
    d, h = displacement.double(), gradient.double()
    hn = h.norm()
    if float(hn) == 0:
        return displacement.clone(), False
    u = h / hn
    a = d.dot(u)
    v = d - a * u
    b = v.norm()
    k = margin / math.sqrt(1 - margin**2)
    if a <= -k * b:
        return displacement.clone(), False
    if b - k * a <= 0:
        return torch.zeros_like(displacement), True
    bp = (b - k * a) / (1 + k*k)
    projected = -k * bp * u + bp / b * v
    assert torch.isfinite(projected).all()
    assert float(projected.dot(h) + margin * hn * projected.norm()) <= 1e-9 * max(1., float(hn * projected.norm()))
    return projected.to(displacement.dtype), True


def geometry_check():
    h = torch.tensor([1., 0.], dtype=torch.float64)
    for d in [torch.tensor(x, dtype=torch.float64) for x in [[1., 1.], [1., 0.], [-1., 0.], [-1., 2.], [0., 0.]]]:
        p, _ = cone_project(d, h)
        assert float(h.dot(p) + .1 * h.norm() * p.norm()) <= 1e-12
        # The exact 2D projection cannot be worse than feasible fixed alternatives.
        for trial in [torch.zeros_like(p), -h, -2*h, torch.tensor([-1., 1.], dtype=torch.float64)]:
            assert float((p-d).square().sum()) <= float((trial-d).square().sum()) + 1e-12
    d = torch.tensor([1., 2.])
    assert torch.equal(cone_project(d, torch.zeros_like(d))[0], d)
    return {'passed': True, 'finite_feasible_nearest_projection_fixtures': 5, 'zero_reference_gradient_identity': True}


def dependency(cfg):
    root = Path(cfg['base_source'])
    base_cfg = json.loads((root / 'CONFIG.private.json').read_text())
    assert base_cfg['source_commit'] == '5c0a106fe77f98b5ae15f04e953ad93cad8831e5'
    name = 'q114_dependency'
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return __import__(name + '.worker', fromlist=['Experiment']).Experiment


def make_experiment(cfg):
    Base = dependency(cfg)

    class Experiment(Base):
        def __init__(self, config):
            super().__init__(config)
            self.phase = 'native'
            self.extra = {'native_RL_updates': 0, 'science_RL_updates': 0,
                          'head_gradient_query_rows': 0, 'head_guard_forward_query_rows': 0}
            self.original_sft = torch.load(Path(config['base_source']) / 'SFT.private.pt', map_location='cpu', weights_only=True)
            self.engine.restore(self.original_sft)
            self.engine.freeze_reference()
            self.arm, self.last_step = None, None

        def flatten(self):
            return torch.cat([p.detach().reshape(-1) for p in self.engine.parameters])

        @torch.no_grad()
        def assign(self, vector):
            lo = 0
            for p in self.engine.parameters:
                p.copy_(vector[lo:lo+p.numel()].view_as(p))
                lo += p.numel()
            assert lo == len(vector)

        def head_loss(self, backward=False):
            total = 0.
            for lo in range(0, len(self.head_labels), 2):
                labels = self.head_labels[lo:lo+2]
                tokens, mask = self.engine.label_tokens([self.names[int(c)] for c in labels])
                lp = self.engine.token_logps(self.head_features[lo:lo+2], tokens, mask)
                loss = -((lp * mask).sum(-1) / mask.sum(-1)).sum() / len(self.head_labels)
                if backward:
                    loss.backward()
                total += float(loss.detach())
            return total

        def prepare_head(self, features, labels):
            self.head_features, self.head_labels = features, labels
            for p in self.engine.parameters:
                p.grad = None
            self.head_loss(backward=True)
            self.head_gradient = torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                            for p in self.engine.parameters]).clone()
            assert torch.isfinite(self.head_gradient).all()
            self.extra['head_gradient_query_rows'] += len(labels)
            for p in self.engine.parameters:
                p.grad = None

        @torch.no_grad()
        def step(self, opt):
            norm = torch.nn.utils.clip_grad_norm_(self.engine.parameters, 1.)
            assert torch.isfinite(norm)
            before = self.flatten()
            opt.step()
            proposed = self.flatten() - before
            candidate, active = cone_project(proposed, self.head_gradient) if self.arm == 'PROJECTED_TARS' else (proposed, False)
            alphas = [1., .5, .25, .125, .0625, 0.]
            losses = []
            for alpha in alphas:
                self.assign(before + alpha * candidate)
                value = self.head_loss()
                assert math.isfinite(value)
                losses.append(value)
            self.extra['head_guard_forward_query_rows'] += len(self.head_labels) * len(alphas)
            baseline = losses[-1]
            choice = next(i for i, value in enumerate(losses) if value <= baseline)
            self.assign(before + alphas[choice] * candidate)
            actual = self.flatten() - before
            self.last_step = {'head_loss_before': baseline, 'head_loss_after': losses[choice],
                              'guard_losses': losses, 'accepted_alpha': alphas[choice],
                              'proposal_L2': float(proposed.norm()), 'candidate_L2': float(candidate.norm()),
                              'applied_L2': float(actual.norm()), 'head_gradient_L2': float(self.head_gradient.norm()),
                              'proposal_head_dot': float(proposed.double().dot(self.head_gradient.double())),
                              'candidate_head_dot': float(candidate.double().dot(self.head_gradient.double())),
                              'actual_head_dot': float(actual.double().dot(self.head_gradient.double())),
                              'cone_active': active, 'nonzero_parameter_update': bool(actual.norm() > 0),
                              'head_guard_verified': losses[choice] <= baseline}
            return float(norm)

        def native(self):
            self.status('NATIVE_ENGINEERING')
            check = geometry_check()
            f, raw = self.engine.encode(self.images(self.train, [0, 1]))
            self.counts['TRAIN_encoder_rows'] += 2
            del raw
            timings, details = [], []
            for arm in ['PROJECTED_TARS', 'SHRINK_TARS']:
                self.arm = arm
                self.engine.restore(self.original_sft)
                opt = self.optimizer(2e-5)
                start = time.time()
                self.prepare_head(f, self.labels[:2])
                rec = self.rl(f, self.labels[:2], opt, torch.ones(23, device='cuda'), 115000)
                torch.cuda.synchronize()
                timings.append(time.time() - start)
                details.append({'arm': arm, 'RL': rec, 'head_guard': self.last_step})
                self.extra['native_RL_updates'] += 1
                self.counts['engineering_optimizer_updates'] += 1
                del opt
            self.engine.restore(self.original_sft)
            restored = self.engine.snapshot()
            assert all(torch.equal(a, b) for first, second in zip(restored, self.original_sft) for a, b in zip(first, second))
            estimate = max(timings) * 4 * 1024 + max(timings) / 16 * 3396 + 1800
            passed = estimate < self.cfg['deadline'] - time.time()
            write(self.out / 'NATIVE_ADMISSION.json', {'passed': passed, 'engineering_only': True, 'TRAIN_images': 2,
                                                     'VAL_TEST': 0, 'geometry': check, 'strict_SFT_restore': True,
                                                     'both_G8_guard_paths': details, 'two_query_seconds': timings,
                                                     'complete_projection_seconds_with_reserve': estimate,
                                                     'remaining_seconds': self.cfg['deadline']-time.time()})
            assert passed, 'FULL_MATRIX_NOT_TIME_ADMITTED'

        def science(self):
            self.phase = 'science'
            root = Path(self.cfg['base_source'])
            assert json.loads((root / 'TRAIN_CACHE_COMPLETE.json').read_text())['rows'] == 6821
            train = np.memmap(root / 'TRAIN_FEATURES.private.bin', mode='r', dtype=np.uint16, shape=(6821,64,2048))
            tape = torch.load(root / 'QUERY_TAPE.private.pt', weights_only=True, map_location='cpu')
            assert tape.shape == (512,8)
            w = torch.tensor(json.loads((root / 'TARS_WEIGHTS.json').read_text())['weights'], device='cuda', dtype=torch.float32)
            head_classes = torch.where(self.class_counts >= 250)[0].tolist()
            assert len(head_classes) == 12
            pools = {c: torch.where(self.labels == c)[0] for c in head_classes}
            generator = torch.Generator().manual_seed(11574005)
            head_tape = []
            for k in range(512):
                classes = [head_classes[int(c)] for c in torch.randint(12,(8,),generator=generator)]
                head_tape.append(torch.stack([pools[c][torch.randint(len(pools[c]),(),generator=generator)] for c in classes]))
            head_tape = torch.stack(head_tape)
            torch.save(head_tape, self.out / 'HEAD_QUERY_TAPE.private.pt')
            self.records = []
            for arm in ['PROJECTED_TARS', 'SHRINK_TARS']:
                self.arm = arm
                self.engine.restore(self.original_sft)
                opt = self.optimizer(2e-5)
                for k, ids in enumerate(tape):
                    self.status('POSTTRAINING', arm=arm, update=k, total=512)
                    start = time.time()
                    self.prepare_head(self.features(train, head_tape[k]), self.labels[head_tape[k]])
                    rec = self.rl(self.features(train, ids), self.labels[ids], opt, w, 11480000+k)
                    self.extra['science_RL_updates'] += 1
                    self.counts['posttraining_optimizer_updates'] += 1
                    rec.update(phase='POSTTRAINING', arm=arm, update=k, seconds=time.time()-start, **self.last_step)
                    self.records.append(rec)
                    if k % 8 == 0:
                        write(self.out / 'TRAINING_CURVES.json', self.records)
                torch.save(self.engine.snapshot(), self.out / f'{arm}.private.pt')
                write(self.out / f'{arm}_SEALED.json', {'updates':512,'epoch':time.time(),'VAL_TEST':0})
                del opt
                gc.collect()
                torch.cuda.empty_cache()
            write(self.out / 'TRAINING_CURVES.json', self.records)
            write(self.out / 'ALL_BRANCHES_SEALED.json', {'arms':['PROJECTED_TARS','SHRINK_TARS'],'epoch':time.time(),'new_VAL_TEST_so_far':0})
            rows = list(csv.DictReader((Path(self.cfg['manifest']) / 'val.csv').open()))
            assert len(rows) == 1698
            assert not {r['identity_component'] for r in rows} & {r['identity_component'] for r in self.train}
            val = np.memmap(root / 'VAL_FEATURES.private.bin', mode='r', dtype=np.uint16, shape=(1698,64,2048))
            results = []
            for arm in ['PROJECTED_TARS','SHRINK_TARS']:
                self.engine.restore(torch.load(self.out / f'{arm}.private.pt', weights_only=True, map_location='cpu'))
                results.append(self.evaluate(val,rows,arm))
            old = json.loads((root / 'FINAL_RESULTS.json').read_text())['results']
            sft = next(r for r in old if r['name']=='SFT')
            main, shrink = results
            gates = {'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,
                     'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
                     'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
                     'macro_over_SHRINK_nonnegative':main['macro']>=shrink['macro'],
                     'tail_over_SHRINK_positive':main['tail']>shrink['tail']}
            write(self.out / 'FINAL_RESULTS.json', {'results':results,'historical_R114_anchors':old,'gates':gates,
                                                   'all_gates_pass':all(gates.values()),'seed':74005,'TEST':0,
                                                   'scope':'static_development_head_protection_diagnostic',
                                                   'independent_confirmation':False,'patient_independence_verified':False})
            self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=all(gates.values()))

    return Experiment(cfg)


def run(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((56 * 2**30 - 128 * 2**20) / total)
    experiment = None
    try:
        experiment = make_experiment(cfg)
        experiment.native()
        experiment.science()
    finally:
        if experiment is not None:
            write(Path(cfg['output']) / 'COSTS.json', {'counts':experiment.counts,'extra':experiment.extra,
                                                     'CPU_seconds':time.process_time()-experiment.cpu,
                                                     'wall_seconds':time.time()-experiment.begin,
                                                     'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                                     'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                                     'R114_training_and_cache_reused_not_newly_executed':True})
