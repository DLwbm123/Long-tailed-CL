"""Bounded actual-stage replay; no reuse of earlier experiments' feature caches."""
import copy
import json
import os
import signal
import time
import traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS, Model, objective, auxiliary, memory_indices
from data import read_rows, Images
from verify import numpy_forward, numpy_loss, numpy_auxiliary, summarize


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def mathematical_check(cfg):
    torch.manual_seed(cfg['seed'])
    means = torch.randn(3,8,dtype=torch.float64)
    model = Model(8,3,means).double()
    with torch.no_grad(): model.up.weight.normal_(0,.1)
    x = torch.randn(12,8,dtype=torch.float64)
    y = torch.tensor([0,0,1,1,2,2]*2)
    target = torch.zeros(12,3,dtype=torch.float64)
    target.scatter_(1,y[:,None],20.)
    valid = torch.ones(12,3,dtype=torch.bool)
    valid[-2:] = False
    history = dict(targets=target,valid=valid)
    nh = dict(targets=target.numpy(),valid=valid.numpy())
    receipts = {}
    for arm in ARMS:
        model.zero_grad()
        loss,_,_ = objective(model,x,y,history,arm,cfg)
        loss.backward()
        weights = {k:v.numpy().astype(np.float64) for k,v in state(model).items()}
        error = abs(float(loss.detach())-numpy_loss(x.numpy(),y.numpy(),nh,weights,arm,cfg))
        errors=[]
        for name,pos in [('head',(0,0)),('up.weight',(2,1)),('down.bias',(0,))]:
            plus={k:v.copy() for k,v in weights.items()};minus={k:v.copy() for k,v in weights.items()}
            plus[name][pos]+=1e-5;minus[name][pos]-=1e-5
            fd=(numpy_loss(x.numpy(),y.numpy(),nh,plus,arm,cfg)-numpy_loss(x.numpy(),y.numpy(),nh,minus,arm,cfg))/2e-5
            errors.append(abs(fd-float(dict(model.named_parameters())[name].grad[pos])))
        perm=torch.randperm(len(y))
        ph={k:v[perm] for k,v in history.items()}
        assert abs(float(objective(model,x[perm],y[perm],ph,arm,cfg)[0].detach())-float(loss.detach()))<1e-10
        assert error<1e-9 and max(errors)<2e-6
        receipts[arm]=dict(objective_error=error,finite_difference_gradient_error=max(errors),permutation_invariant=True,synthetic_torch_objectives=2,synthetic_numpy_objectives=7)
    logits=torch.tensor([[2.,0.,4.]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[2.,0.,0.]],dtype=torch.float64);valid=torch.tensor([[True,True,False]]);y=torch.tensor([0])
    logit_value=auxiliary(logits,y,target,valid,'LOGIT',25.)
    margin_value=auxiliary(logits,y,target,valid,'MARGIN',25.)
    margin_value.backward()
    assert float(logit_value.detach())==0 and float(margin_value.detach())>0 and logits.grad[0,2]>0 and logits.grad[0,0]<0
    wrong=target.clone();wrong[0,:2]=torch.tensor([0.,2.],dtype=torch.float64)
    assert float(auxiliary(logits,y,wrong,valid,'MARGIN',25.).detach())==0
    receipts['new_competitor_counterexample']=dict(historical_logit_loss=float(logit_value.detach()),margin_loss=float(margin_value.detach()),new_logit_descent_direction=True,wrong_teacher_ignored=True,scope='Synthetic algebra only; no NIH performance or proof of unique failure cause.')
    return receipts


def main(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(cfg['output'])
    root.mkdir(exist_ok=False)
    cost = dict(qualification_encodes=0, science_train_encodes=0, validation_encodes=0,
                qualification_updates=0, qualification_model_rows={a: 0 for a in ARMS},
                qualification_FP64_model_rows={a: 0 for a in ARMS}, qualification_numpy_rows={a: 0 for a in ARMS},
                updates={a: 0 for a in ARMS}, sampled_image_rows={a: 0 for a in ARMS},
                training_view_rows={a: 0 for a in ARMS}, validation_model_rows={a: 0 for a in ARMS},
                encoder_parameter_updates=0, encoder_seconds={}, contrastive_similarity_pairs={a: 0 for a in ARMS},
                training_probe_model_rows={a:0 for a in ARMS}, birth_target_model_rows={a:0 for a in ARMS}, birth_target_logit_elements={a:0 for a in ARMS},
                functional_objective_model_rows={a:0 for a in ARMS}, historical_target_elements={a:0 for a in ARMS},
                allseen_margin_rival_elements={a:0 for a in ARMS}, history_memory_rows_final=0)
    models, optimizers, curves, seals = {}, {}, [], []

    def status(phase, **kwargs):
        cost.update(wall_from_first=time.time()-cfg['original_start_epoch'],
                    GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root / 'COSTS.private.json', cost)
        write(root / 'status.private.json', dict(state='RUNNING', phase=phase, updated=time.time(), cost=cost, **kwargs))
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        assert cost['GPU_peak_reserved_bytes'] < cfg['gpu_reserved_budget_bytes']

    def interrupt(sig, frame):
        save(root / 'interrupted.private.pt', dict(models={a: state(m) for a, m in models.items()},
             optimizers={a: o.state_dict() for a, o in optimizers.items()}, cost=cost))
        write(root / 'INTERRUPTED.private.json', dict(at=time.time(), cost=cost))
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupt)
    status('mathematical_qualification')
    math_receipt = mathematical_check(cfg)
    write(root / 'MATH_QUALIFICATION.json', math_receipt)
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    for i, row in enumerate(train):
        row['id'] = i
    assert len(train) == 68058 and len(val) == 300
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    write(root / 'COVERAGE.json', dict(train_counts=dict(Counter(str(r['label']) for r in train)),
          patient_overlap=0, base_images=sum(r['label'] in cfg['order'][:10] for r in train),
          total_train=68058, total_val=300, new_feature_memory_permission=True,
          historical_image_replay=False, reconstruction_of_deleted_cache=False))
    import timm
    from safetensors.torch import load_file
    status('encoder_loading')
    encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    encoder.load_state_dict({k: v for k, v in load_file(cfg['weight']).items() if not k.startswith('head.')}, strict=True)
    encoder.eval().requires_grad_(False).cuda()

    def extract(rows, field):
        out = torch.empty(len(rows), cfg['input_dimension'], dtype=torch.float32)
        tick = time.time()
        for images, idx in DataLoader(Images(rows, cfg), batch_size=32, num_workers=0, shuffle=False):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                raw = encoder.forward_features(images.cuda())
            assert raw.shape[1:] == (197, 768) and torch.isfinite(raw).all()
            raw = raw.float()
            out[idx] = F.normalize(torch.cat([raw[:, 0], raw[:, 1:].mean(1)], 1), dim=1).cpu()
            torch.cuda.synchronize()
            cost[field] += len(idx)
            cost['encoder_seconds'][field] = cost['encoder_seconds'].get(field, 0) + time.time() - tick
            status(field, encoded_in_current_set=int(idx[-1])+1, current_count=len(rows))
            tick = time.time()
        return out

    qrows = []
    for c in cfg['order'][:10]:
        groups = set()
        for r in train:
            if r['label'] == c and r['group'] not in groups:
                qrows.append(r)
                groups.add(r['group'])
                if len(groups) == 2:
                    break
        assert len(groups) == 2
    q = extract(qrows, 'qualification_encodes')
    torch.manual_seed(cfg['seed'])
    qmeans = torch.stack([q[2*i:2*i+2].mean(0) for i in range(10)])
    template = Model(cfg['input_dimension'], cfg['adapter_rank'], qmeans)
    views = q.repeat(2, 1)
    gen = torch.Generator().manual_seed(cfg['seed']+2)
    views *= (torch.rand(views.shape, generator=gen) >= cfg['input_dropout']) / (1-cfg['input_dropout'])
    y = torch.arange(10).repeat_interleave(2).repeat(2)
    with torch.no_grad():
        target = template(views, cfg['logit_scale'])[1].detach().clone()
    cost['qualification_teacher_rows']=len(views)
    qhistory = dict(targets=target,valid=torch.ones_like(target,dtype=torch.bool))
    qualification = {}
    for arm in ARMS:
        model = copy.deepcopy(template)
        loss, ce, aux = objective(model, views, y, qhistory, arm, cfg)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0
        cost['qualification_model_rows'][arm] += len(views)
        s = {k: v.numpy().astype(np.float64) for k, v in state(model).items()}
        ref = numpy_forward(views.numpy(), s, cfg['logit_scale'])[1]
        with torch.no_grad():
            fp64 = copy.deepcopy(model).double()(views.double(), cfg['logit_scale'])[1]
        error = float(np.max(abs(ref-fp64.numpy())))
        assert error < cfg['score_absolute_tolerance']
        cost['qualification_FP64_model_rows'][arm] += len(views)
        cost['qualification_numpy_rows'][arm] += len(views)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        opt.step()
        cost['qualification_updates'] += 1
        qualification[arm] = dict(loss=float(loss.detach()), CE=float(ce.detach()), auxiliary=float(aux.detach()),
                                  score_error=error, engineering_updates=1)
    write(root / 'QUALIFICATION.json', dict(state='PASS', math=math_receipt, real=qualification,
          real_training_encodes=20, engineering_updates=3, validation_encodes=0,
          timing_is_not_formal_throughput=True))
    del model, template, q, qrows, qmeans, opt, views, loss, ce, aux, fp64
    torch.cuda.empty_cache()
    memory_features = torch.empty(0, cfg['input_dimension'])
    memory_rows = []
    memory_targets={a:torch.empty(0,20,dtype=torch.float64) for a in ARMS}
    memory_valid=torch.empty(0,20,dtype=torch.bool)
    for task in range(6):
        seen = cfg['order'][:sum(cfg['task_sizes'][:task+1])]
        new = seen[sum(cfg['task_sizes'][:task]):]
        rows = [r for r in train if r['label'] in new]
        features = extract(rows, 'science_train_encodes')
        means = []
        for c in new:
            patients = {}
            for i, r in enumerate(rows):
                if r['label'] == c:
                    patients.setdefault(r['group'], []).append(i)
            means.append(torch.stack([features[ix].double().mean(0) for ix in patients.values()]).mean(0).float())
        means = torch.stack(means)
        if task == 0:
            torch.manual_seed(cfg['seed'])
            template = Model(cfg['input_dimension'], cfg['adapter_rank'], means)
            models = {a: copy.deepcopy(template).cuda() for a in ARMS}
            del template
        else:
            for m in models.values():
                m.expand(means.cuda())
        optimizers = {a: torch.optim.AdamW(m.parameters(), lr=cfg['learning_rate'],
                      weight_decay=cfg['weight_decay']) for a, m in models.items()}
        # Old data consist only of explicit bounded memory, current data of this arrival.
        source = torch.cat([memory_features, features])
        source_rows = memory_rows + rows
        pools = {}
        for i, r in enumerate(source_rows):
            pools.setdefault(r['label'], {}).setdefault(r['group'], []).append(i)
        assert set(pools) == set(seen)
        rng = np.random.default_rng(cfg['seed'] + 1000 + task)
        view_rng = torch.Generator().manual_seed(cfg['seed'] + 2000 + task)
        steps = cfg['base_updates'] if task == 0 else cfg['later_updates']
        step_start = time.time()
        for step in range(1, steps+1):
            chosen_classes = rng.choice(seen, size=cfg['classes_per_batch'], replace=False).tolist()
            indices = []
            for c in chosen_classes:
                gs = list(pools[c])
                for _ in range(cfg['draws_per_class']):
                    g = gs[int(rng.integers(len(gs)))]
                    values = pools[c][g]
                    indices.append(values[int(rng.integers(len(values)))])
            x = source[indices].repeat(2, 1)
            x *= (torch.rand(x.shape, generator=view_rng) >= cfg['input_dropout']) / (1-cfg['input_dropout'])
            x = x.cuda()
            y = torch.tensor([seen.index(source_rows[i]['label']) for i in indices]*2, device='cuda')
            draw = indices*2
            old = torch.tensor([i<len(memory_rows) for i in draw],device='cuda')
            valid = torch.zeros(len(draw),len(seen),dtype=torch.bool,device='cuda')
            if bool(old.any()):
                memory_ix=torch.tensor([i for i in draw if i<len(memory_rows)])
                valid[old]=memory_valid[memory_ix,:len(seen)].cuda()
            details = {}
            for a in ARMS:
                opt = optimizers[a]
                opt.zero_grad()
                target = torch.zeros(len(draw),len(seen),device='cuda')
                if bool(old.any()): target[old]=memory_targets[a][memory_ix,:len(seen)].float().cuda()
                history=dict(targets=target,valid=valid)
                loss, ce, aux = objective(models[a], x, y, history, a, cfg)
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(models[a].parameters(), cfg['gradient_clip'])
                assert torch.isfinite(norm)
                opt.step()
                cost['updates'][a] += 1
                cost['sampled_image_rows'][a] += len(indices)
                cost['training_view_rows'][a] += len(x)
                cost['functional_objective_model_rows'][a]+=len(x)
                if a == 'LOGIT': cost['historical_target_elements'][a]+=int(valid.sum())
                if a == 'MARGIN':
                    rivals=valid.clone();rivals.scatter_(1,y[:,None],False)
                    birth=target.gather(1,y[:,None]).squeeze(1)-target.masked_fill(~rivals,-torch.inf).max(1).values
                    cost['allseen_margin_rival_elements'][a]+=int((old & (birth>0)).sum())*(len(seen)-1)
                details[a] = dict(loss=float(loss.detach()), CE=float(ce.detach()), auxiliary=float(aux.detach()),
                                  gradient_norm_before_clip=float(norm))
            if step == 1 or step % 100 == 0 or step == steps:
                torch.cuda.synchronize()
                curves.append(dict(task=task, step=step, arms=details, at=time.time(),
                                   elapsed_stage_training=time.time()-step_start))
                write(root / 'TRAINING_CURVES.json', curves)
                status('actual_stage_training', task=task, step=step, steps=steps, seen=seen)
        probe=dict(x=x.detach().cpu().double(),y=y.detach().cpu(),history={},losses={})
        for a in ARMS:
            pt=torch.zeros(len(draw),len(seen))
            if bool(old.any()): pt[old.cpu()]=memory_targets[a][memory_ix,:len(seen)].float()
            ph=dict(targets=pt.double(),valid=valid.cpu())
            pm=copy.deepcopy(models[a]).cpu().double()
            with torch.no_grad(): pl=objective(pm,probe['x'],probe['y'],ph,a,cfg)[0]
            probe['history'][a]=ph;probe['losses'][a]=float(pl)
            cost['training_probe_model_rows'][a]+=len(x)
            del pm
        selected = memory_indices(rows, cfg['memory_per_class'], cfg['seed'])
        selected_features = features[selected]
        born_valid=torch.zeros(len(selected),20,dtype=torch.bool);born_valid[:,:len(seen)]=True
        for a in ARMS:
            teacher=copy.deepcopy(models[a]).cpu().double()
            with torch.no_grad():
                _, birth=teacher(selected_features.double(),cfg['logit_scale'])
            pad=torch.zeros(len(selected),20,dtype=torch.float64)
            pad[:,:len(seen)]=birth
            del teacher
            memory_targets[a]=torch.cat([memory_targets[a],pad])
            cost['birth_target_model_rows'][a]+=len(selected)
            cost['birth_target_logit_elements'][a]+=birth.numel()
        memory_valid=torch.cat([memory_valid,born_valid])
        memory_features = torch.cat([memory_features, selected_features])
        cost['history_memory_rows_final']=len(memory_features)
        memory_rows.extend(rows[i] for i in selected)
        assert len(memory_rows) <= len(seen) * cfg['memory_per_class']
        assert len(memory_rows) == len({(r['label'], r['group']) for r in memory_rows})
        save(root / f'stage_{task}.private.pt', dict(seen=seen, models={a: state(m) for a, m in models.items()},
             memory_features=memory_features, memory_rows=memory_rows, memory_targets=memory_targets, memory_valid=memory_valid, training_probe=probe, new_head_initialization_means=means))
        seals.append(dict(task=task, at=time.time(), seen=seen, validation_encodes=0,
                          science_train_encodes=cost['science_train_encodes'], updates=dict(cost['updates']),
                          memory_images=len(memory_rows), parameters_per_arm=sum(p.numel() for p in models['ER'].parameters())))
        write(root / 'STAGE_SEALS.json', seals)
        del source, source_rows, features, rows, pools, x, y, loss, ce, aux
        torch.cuda.empty_cache()
    assert cost['science_train_encodes'] == 68058
    write(root / 'FIT_SEAL.json', dict(at=time.time(), models=18, stages=6, validation_encodes=0))
    write(root / 'QUERY_OPEN.private.json', dict(at=time.time(), validation_encodes=0))
    query = extract(val, 'validation_encodes')
    labels = torch.tensor([r['label'] for r in val])
    save(root / 'query.private.pt', dict(features=query, labels=labels))
    stages = []
    for task in range(6):
        pack = torch.load(root / f'stage_{task}.private.pt', map_location='cpu', weights_only=False)
        seen = pack['seen']
        mask = torch.isin(labels, torch.tensor(seen))
        metrics = {}
        for a in ARMS:
            m = Model(cfg['input_dimension'], cfg['adapter_rank'], pack['models'][a]['head']).double()
            m.load_state_dict({k: v.double() for k, v in pack['models'][a].items()})
            with torch.inference_mode():
                _, score = m(query[mask].double(), cfg['logit_scale'])
            cost['validation_model_rows'][a] += int(mask.sum())
            save(root / f'scores_{task}_{a}.private.pt', score)
            pred = torch.tensor(seen)[score.argmax(1)]
            correct = {str(c): int(((pred == c) & (labels[mask] == c)).sum()) for c in seen}
            metrics[a] = dict(correct=correct, total_correct=sum(correct.values()),
                              BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task, seen=seen, metrics=metrics))
        status('sealed_model_scoring', task=task)
    result, decision = summarize(stages, cfg, Counter(r['label'] for r in train))
    write(root / 'RESULTS.json', result)
    write(root / 'DECISION.json', decision)
    status('scientific_complete')
    write(root / 'COSTS.json', cost)
    write(root / 'status.private.json', dict(state='COMPLETE_PENDING_VERIFY', updated=time.time(), cost=cost))


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:
        main(cfg)
    except BaseException as error:
        root = Path(cfg['output'])
        if root.exists():
            write(root / 'FAILURE.private.json', dict(at=time.time(), error=repr(error), traceback=traceback.format_exc()))
        raise
