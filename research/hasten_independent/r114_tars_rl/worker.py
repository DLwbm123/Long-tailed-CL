"""One prospective bootstrap, matched recognition posttraining, then sealed VAL."""
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .core import Engine, advantages, normalize, software_check, surrogate, tailness, write


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg['output'])
        self.counts = {'engineering_optimizer_updates': 0, 'SFT_updates': 0,
                       'posttraining_optimizer_updates': 0, 'generation_responses': 0,
                       'TRAIN_encoder_rows': 0, 'VAL_encoder_rows': 0}
        self.begin = time.time()
        self.cpu = time.process_time()
        self.records = []
        self.train = list(csv.DictReader((Path(cfg['manifest']) / 'train.csv').open()))
        assert len(self.train) == len({r['identity_component'] for r in self.train}) == 6821
        self.names = {}
        for r in self.train:
            c = int(r['original_label'])
            name = r['official_class_name'].replace('-', ' ')
            assert self.names.get(c, name) == name
            self.names[c] = name
        assert set(self.names) == set(range(23)) and len({normalize(s) for s in self.names.values()}) == 23
        self.labels = torch.tensor([int(r['original_label']) for r in self.train])
        self.class_counts = torch.bincount(self.labels, minlength=23)
        self.engine = Engine(cfg)
        for s in self.names.values():
            self.engine.label_tokens([s])

    def guard(self):
        assert time.time() < self.cfg['deadline'] - 120, 'ORIGINAL_CLOCK'
        assert torch.cuda.memory_reserved() < 56 * 2**30, 'GPU_ALLOCATION_LIMIT'

    def status(self, phase, **kw):
        self.guard()
        write(self.out / 'STATUS.json', {'phase': phase, 'epoch': time.time(), 'counts': self.counts, **kw})

    def images(self, rows, ids):
        out = []
        for i in ids:
            with Image.open(Path(self.cfg['images']) / rows[int(i)]['relative_path']) as im:
                out.append(im.convert('RGB'))
        return out

    def optimizer(self, lr):
        return torch.optim.AdamW(self.engine.parameters, lr=lr, weight_decay=0)

    def step(self, opt):
        norm = torch.nn.utils.clip_grad_norm_(self.engine.parameters, 1.)
        assert torch.isfinite(norm), 'NONFINITE_GRADIENT'
        opt.step()
        return float(norm)

    def ce(self, features, labels, opt, weights):
        opt.zero_grad(set_to_none=True)
        total = 0.
        for lo in range(0, len(labels), 2):
            y = labels[lo:lo+2]
            tok, mask = self.engine.label_tokens([self.names[int(c)] for c in y])
            lp = self.engine.token_logps(features[lo:lo+2], tok, mask)
            loss = -((lp * mask).sum(-1) / mask.sum(-1) * weights[y.to(weights.device)]).sum() / len(labels)
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        return total, self.step(opt)

    def rl(self, features, labels, opt, weights, draw_seed):
        n, G = len(labels), 8
        g = torch.Generator(device='cuda').manual_seed(draw_seed)
        f = features.repeat_interleave(G, 0)
        tok, mask, sampled_lp, text, done = self.engine.generate(f, generator=g, sample=True)
        target = [normalize(self.names[int(c)]) for c in labels for _ in range(G)]
        reward = torch.tensor([int(t == y and bool(d)) for t, y, d in zip(text, target, done)], device='cuda', dtype=torch.float32).reshape(n, G)
        adv = advantages(reward).flatten().detach()
        # Reference evaluations do not alter trainable adapter storage or optimizer state.
        refs = []
        with torch.no_grad(), self.engine.reference():
            for lo in range(0, n * G, 2):
                refs.append(self.engine.token_logps(f[lo:lo+2], tok[lo:lo+2], mask[lo:lo+2]))
        ref = torch.cat(refs).detach()
        # BF16 cached decoding and full-sequence training use different GEMM shapes.
        # Freeze the behavior forward in exactly the optimization microbatch path,
        # so an unchanged policy starts at ratio1 rather than numerical clipping.
        with torch.no_grad():
            old = torch.cat([self.engine.token_logps(f[lo:lo+2], tok[lo:lo+2], mask[lo:lo+2])
                             for lo in range(0, n * G, 2)]).detach()
        opt.zero_grad(set_to_none=True)
        loss_sum, pg_sum, kl_sum, margin = 0., 0., 0., 0.
        w = weights[labels.to(weights.device)].repeat_interleave(G)
        for lo in range(0, n * G, 2):
            lp = self.engine.token_logps(f[lo:lo+2], tok[lo:lo+2], mask[lo:lo+2])
            m = mask[lo:lo+2]
            margin = max(margin, float(((lp.detach() - old[lo:lo+2]).abs() * m).max()))
            loss, pg, kl = surrogate(lp, old[lo:lo+2], ref[lo:lo+2], m, adv[lo:lo+2], w[lo:lo+2])
            (loss * len(lp) / (n * G)).backward()
            loss_sum += float(loss.detach()) * len(lp) / (n * G)
            pg_sum += float(pg.sum()) / (n * G)
            kl_sum += float(kl.sum()) / (n * G)
        assert margin == 0., ('UNCHANGED_POLICY_RATIO_NOT_ONE', margin)
        norm = self.step(opt)
        self.counts['generation_responses'] += n * G
        constant = reward.std(-1, unbiased=False) == 0
        return {'loss': loss_sum, 'surrogate_unweighted': pg_sum, 'KL': kl_sum,
                'gradient_norm': norm, 'reward_mean': float(reward.mean()),
                'constant_groups': int(constant.sum()), 'groups': n,
                'correct_by_class': torch.bincount(labels, weights=reward.sum(-1).cpu(), minlength=23).tolist(),
                'responses_by_class': (torch.bincount(labels, minlength=23) * G).tolist(),
                'truncated_responses': int((~done).sum()), 'tokens': int(mask.sum()),
                'unchanged_policy_logp_max_difference': margin,
                'cached_sampling_vs_behavior_forward_max_difference': float(((sampled_lp - old).abs() * mask).max()),
                'all_advantages_zero': bool((adv == 0).all())}

    def native(self):
        self.status('NATIVE_ENGINEERING')
        check = software_check()
        initial = self.engine.snapshot()
        start = time.time()
        features, raw = self.engine.encode(self.images(self.train, [0, 1]))
        torch.cuda.synchronize()
        enc = time.time() - start
        self.counts['TRAIN_encoder_rows'] += 2
        with torch.no_grad():
            native = self.engine.model(**raw, use_cache=False).logits[:, -1].float()
            ids = self.engine.prompt_ids.expand(2, -1)
            pos, _ = self.engine.positions(ids, torch.ones_like(ids))
            hidden = self.engine.model.model(inputs_embeds=self.engine.embeddings(ids, features), position_ids=pos,
                                            attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).last_hidden_state
            cached = self.engine.model.lm_head(hidden[:, -1]).float()
            parity = float((native - cached).abs().max())
            assert parity <= .05, ('NATIVE_CACHE_PARITY', parity)
        y = self.labels[:2]
        weights = torch.ones(23, device='cuda')
        opt = self.optimizer(2e-4)
        t = time.time()
        value, norm = self.ce(features, y, opt, weights)
        torch.cuda.synchronize()
        ce_seconds = time.time() - t
        self.counts['engineering_optimizer_updates'] += 1
        self.engine.freeze_reference()
        t = time.time()
        info = self.rl(features, y, self.optimizer(2e-5), weights, 114000)
        torch.cuda.synchronize()
        rl_seconds = time.time() - t
        self.counts['engineering_optimizer_updates'] += 1
        state = self.engine.snapshot()
        save = self.out / 'NATIVE_ADAPTER.private.pt'
        torch.save(state, save)
        self.engine.restore(initial)
        self.engine.restore(torch.load(save, map_location='cpu', weights_only=True))
        assert all(torch.equal(x, y) for pair, other in zip(state, self.engine.snapshot()) for x, y in zip(pair, other))
        self.engine.restore(initial)
        self.engine.freeze_reference()
        # Native generation uses the full24-token cap; count all planned work and reserve.
        projected = enc / 2 * (6821 + 1698) + ce_seconds / 2 * (2 * 6821 + 2 * 512 * 8)
        projected += rl_seconds / 2 * (3 * 512 * 8)
        # TRAIN greedy plus six VAL readouts cost at most full sampled generation time.
        projected += rl_seconds / 16 * (6821 + 6 * 1698) + 3600
        admitted = projected < self.cfg['deadline'] - time.time()
        receipt = {'passed': admitted, 'engineering_only': True, 'parents': 2, 'VAL_TEST': 0,
                   'cache_native_max_abs_logit_error': parity, 'adapter_strict_restore': True,
                   'software': check, 'CE_two_queries_seconds': ce_seconds, 'RL_two_G8_queries_seconds': rl_seconds,
                   'visual_two_queries_seconds': enc, 'complete_projection_seconds_including_reserve': projected,
                   'remaining_seconds': self.cfg['deadline'] - time.time(), 'native_CE_loss': value,
                   'native_CE_gradient_norm': norm, 'native_RL': info, 'counts': dict(self.counts)}
        write(self.out / 'NATIVE_ADMISSION.json', receipt)
        del opt, raw, initial, state, features, cached, native, hidden
        gc.collect()
        torch.cuda.empty_cache()
        assert admitted, 'FULL_MATRIX_NOT_TIME_ADMITTED'

    def cache(self, rows, split):
        path = self.out / f'{split}_FEATURES.private.bin'
        data = np.memmap(path, mode='w+', dtype=np.uint16, shape=(len(rows), 64, 2048))
        for lo in range(0, len(rows), 8):
            self.status('ENCODING_' + split, rows_done=lo, rows_total=len(rows))
            ids = list(range(lo, min(lo+8, len(rows))))
            f, _ = self.engine.encode(self.images(rows, ids))
            data[lo:lo+len(ids)] = f.detach().cpu().view(torch.uint16).numpy()
            self.counts[split + '_encoder_rows'] += len(ids)
        data.flush()
        return data

    def features(self, data, ids):
        return torch.from_numpy(np.asarray(data[np.asarray(ids)]).copy()).view(torch.bfloat16).cuda()

    def sft(self, data):
        self.status('COMMON_SFT')
        opt = self.optimizer(2e-4)
        total = 2 * math.ceil(6821 / 16)
        g = torch.Generator().manual_seed(74005)
        for epoch in range(2):
            indices = torch.randperm(6821, generator=g)
            for lo in range(0, 6821, 16):
                k = self.counts['SFT_updates']
                self.status('COMMON_SFT', training_epoch=epoch, update=k, total=total)
                opt.param_groups[0]['lr'] = 2e-4 * .5 * (1 + math.cos(math.pi * k / total))
                ids = indices[lo:lo+16]
                loss, norm = self.ce(self.features(data, ids), self.labels[ids], opt, torch.ones(23, device='cuda'))
                self.counts['SFT_updates'] += 1
                self.records.append({'phase': 'SFT', 'epoch': epoch, 'update': k, 'loss': loss, 'gradient_norm': norm})
                if k % 32 == 0:
                    write(self.out / 'TRAINING_CURVES.json', self.records)
        state = self.engine.snapshot()
        torch.save(state, self.out / 'SFT.private.pt')
        self.engine.freeze_reference()
        del opt
        return state

    @torch.no_grad()
    def evaluate(self, data, rows, name):
        labels = torch.tensor([int(r['original_label']) for r in rows])
        correct, den = torch.zeros(23, dtype=torch.long), torch.bincount(labels, minlength=23)
        assert bool((den > 0).all())
        predictions, truncated = [], 0
        lookup = {normalize(v): k for k, v in self.names.items()}
        for lo in range(0, len(rows), 16):
            self.status('GREEDY_' + name, rows_done=lo, rows_total=len(rows))
            ids = list(range(lo, min(lo+16, len(rows))))
            _, mask, _, text, done = self.engine.generate(self.features(data, ids))
            predicted = [lookup.get(t, -1) if bool(d) else -1 for t, d in zip(text, done)]
            p = torch.tensor(predicted)
            predictions.extend(predicted)
            truncated += int((~done).sum())
            good = labels[ids] == p
            correct += torch.bincount(labels[ids][good], minlength=23)
            self.counts['generation_responses'] += len(ids)
        rate = correct.double() / den
        result = {'name': name, 'correct': correct.tolist(), 'denominator': den.tolist(),
                  'micro': float(correct.sum() / den.sum()), 'macro': float(rate.mean()),
                  'head': float(rate[self.class_counts >= 250].mean()),
                  'tail': float(rate[self.class_counts < 250].mean()),
                  'tiny': float(rate[self.class_counts < 25].mean()),
                  'invalid_predictions': sum(p < 0 for p in predictions), 'truncated': truncated}
        write(self.out / f'{name}_COUNTS.json', result)
        torch.save({'predictions': torch.tensor(predictions), 'labels': labels}, self.out / f'{name}_PREDICTIONS.private.pt')
        return result

    def science(self):
        existing = self.out / 'TRAIN_CACHE_COMPLETE.json'
        if existing.exists():
            receipt = json.loads(existing.read_text())
            assert receipt['rows'] == 6821 and receipt['features_shape'] == [6821, 64, 2048]
            train = np.memmap(self.out / 'TRAIN_FEATURES.private.bin', mode='r', dtype=np.uint16, shape=(6821, 64, 2048))
        else:
            train = self.cache(self.train, 'TRAIN')
            write(existing, {'rows': 6821, 'features_shape': [6821, 64, 2048], 'dtype': 'BF16', 'TRAIN_only': True})
        sft = self.sft(train)
        fit = self.evaluate(train, self.train, 'SFT_TRAIN')
        w, constant, median = tailness(fit['denominator'], fit['correct'])
        write(self.out / 'TARS_WEIGHTS.json', {'class_names': self.names, 'counts': fit['denominator'], 'correct': fit['correct'],
                                             'weights': w.tolist(), 'constant_strength': constant, 'raw_median': median,
                                             'error_source': 'fixed_SFT_TRAIN_greedy_once', 'VAL_TEST': 0})
        g = torch.Generator().manual_seed(11474005)
        tape = torch.randint(6821, (512, 8), generator=g)
        torch.save(tape, self.out / 'QUERY_TAPE.private.pt')
        for arm in ['CE_CONT', 'TARS_CE', 'GRPO', 'CONST_GRPO', 'TARS_GRPO']:
            self.engine.restore(sft)
            opt = self.optimizer(2e-5)
            weights = (w if arm.startswith('TARS') else torch.full((23,), constant) if arm == 'CONST_GRPO' else torch.ones(23)).float().cuda()
            for k, ids in enumerate(tape):
                self.status('POSTTRAINING', arm=arm, update=k, total=512)
                t = time.time()
                features, labels = self.features(train, ids), self.labels[ids]
                if arm in ['CE_CONT', 'TARS_CE']:
                    loss, norm = self.ce(features, labels, opt, weights)
                    rec = {'loss': loss, 'gradient_norm': norm, 'groups': 0}
                else:
                    rec = self.rl(features, labels, opt, weights, 11480000 + k)
                self.counts['posttraining_optimizer_updates'] += 1
                rec.update(phase='POSTTRAINING', arm=arm, update=k, seconds=time.time()-t)
                self.records.append(rec)
                if k % 8 == 0:
                    write(self.out / 'TRAINING_CURVES.json', self.records)
            torch.save(self.engine.snapshot(), self.out / f'{arm}.private.pt')
            write(self.out / f'{arm}_SEALED.json', {'arm': arm, 'updates': 512, 'epoch': time.time(), 'VAL_TEST': 0})
            del opt, features
            gc.collect()
            torch.cuda.empty_cache()
        write(self.out / 'ALL_BRANCHES_SEALED.json', {'epoch': time.time(), 'arms': ['SFT', 'CE_CONT', 'TARS_CE', 'GRPO', 'CONST_GRPO', 'TARS_GRPO'], 'VAL_TEST_so_far': 0})
        # VAL is not opened or encoded until the entire prospective matrix is sealed.
        val_rows = list(csv.DictReader((Path(self.cfg['manifest']) / 'val.csv').open()))
        assert len(val_rows) == 1698
        assert not {r['identity_component'] for r in val_rows} & {r['identity_component'] for r in self.train}
        val = self.cache(val_rows, 'VAL')
        results = []
        for arm in ['SFT', 'CE_CONT', 'TARS_CE', 'GRPO', 'CONST_GRPO', 'TARS_GRPO']:
            self.engine.restore(torch.load(self.out / f'{arm}.private.pt', weights_only=True, map_location='cpu'))
            results.append(self.evaluate(val, val_rows, arm))
        by = {r['name']: r for r in results}
        tars, base = by['TARS_GRPO'], by['SFT']
        gates = {'macro_over_SFT_ge_1pp': tars['macro'] - base['macro'] >= .01,
                 'tail_over_SFT_ge_2pp': tars['tail'] - base['tail'] >= .02,
                 'head_over_SFT_ge_minus1pp': tars['head'] - base['head'] >= -.01}
        for other in ['GRPO', 'CONST_GRPO']:
            for metric in ['macro', 'tail']:
                gates[f'{metric}_over_{other}_positive'] = tars[metric] > by[other][metric]
        gates['macro_over_TARS_CE_nonnegative'] = tars['macro'] >= by['TARS_CE']['macro']
        gates['tail_over_TARS_CE_positive'] = tars['tail'] > by['TARS_CE']['tail']
        rescue = {}
        main = torch.load(self.out / 'TARS_GRPO_PREDICTIONS.private.pt', weights_only=True)
        for other in by:
            saved = torch.load(self.out / f'{other}_PREDICTIONS.private.pt', weights_only=True)
            a, b = main['predictions'] == main['labels'], saved['predictions'] == saved['labels']
            rescue[other] = {'rescued': int((a & ~b).sum()), 'harmed': int((~a & b).sum())}
        write(self.out / 'FINAL_RESULTS.json', {'scope': 'static_HK23_development_generative_RL_diagnostic',
                                               'results': results, 'gates': gates, 'all_gates_pass': all(gates.values()),
                                               'rescue_harm': rescue, 'seed': 74005, 'TEST': 0,
                                               'patient_independence_verified': False, 'independent_confirmation': False})
        write(self.out / 'TRAINING_CURVES.json', self.records)
        self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION', all_gates_pass=all(gates.values()))


def run(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((56 * 2**30 - 128 * 2**20) / total)
    exp = None
    try:
        exp = Experiment(cfg)
        exp.native()
        exp.science()
    finally:
        if exp is not None:
            write(Path(cfg['output']) / 'COSTS.json', {'counts': exp.counts, 'CPU_seconds': time.process_time()-exp.cpu,
                                                     'wall_seconds': time.time()-exp.begin,
                                                     'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                                     'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved()})


if __name__ == '__main__':
    run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
