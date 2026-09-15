"""One real update, exact frozen-state and resume/continuation checks."""
import copy
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_medical_v2 import write_json,network_hash,rng_equal
from stationary_medical_v4 import fork,HEADS
from diagnose_medical_v3 import fixed_probe,batch_probe,gradient_probe,isolated_rng
from utils.medical_v2 import effective_optimizer

def engineering(config):
    out=Path(config['output'])/'engineering';out.mkdir(parents=True,exist_ok=True)
    l=fork(config,1993,out/'scratch');l._network.cuda()
    original=network_hash(l._network);l.args['tuned_epoch']=1
    x,y=fixed_probe(l);x=x[:4];y=y[:4]
    batch=[(torch.arange(len(x)),x,y)]
    losses=[];grad={};inject_moments=[False]
    def capture(epoch,batch,loss):
        for name,p in l._network.named_parameters():
            if p.grad is not None:grad[name]=p.grad.detach().cpu().clone()
        assert set(grad)==HEADS and all(torch.isfinite(g).all() for g in grad.values())
        if inject_moments[0]:
            for group in l.optimizer.param_groups:
                for p in group['params']:
                    for key in ('exp_avg','exp_avg_sq'):l.optimizer.state[p][key][:4].fill_(0.25)
            inject_moments[0]=False
    l._v2_gradient_hook=capture
    l._v2_epoch_hook=lambda epoch,opt,sched,stats:losses.append(stats['loss'])
    def update():
        l._v2_start_epoch=0
        l._init_train(batch,None,l.optimizer,None)
    l.optimizer=effective_optimizer(l._network.backbone)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();update();torch.cuda.synchronize()
    seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()
    l.assert_stationary();assert network_hash(l._network)!=original
    row_fixed=config.get('old_classifier_rows')=='session_fixed'
    if row_fixed:
        l.assert_old_rows(l.optimizer)
        assert l.old_row_reference['steps']==1
        for name in HEADS:assert torch.equal(l.old_row_reference['rows'][name],l.base_network[name][:4])
        assert any(not torch.equal(p[4:6].detach().cpu(),l.base_network[n][4:6]) for n,p in l._network.named_parameters() if n in HEADS)
        # Stress nonzero inherited moments; the post-step projection must erase them.
        inject_moments[0]=True
    assert all((g[6:]==0).all() for g in grad.values())
    if row_fixed:
        # This update also exercises nonzero weight decay in the original AdamW groups.
        l._init_train(batch,None,l.optimizer,None);l.assert_old_rows(l.optimizer)
        losses.clear()
    ckpt=out/'roundtrip.pt';l.checkpoint(ckpt,l.optimizer,None,1,'epoch_complete')
    h=network_hash(l._network);rng=l._capture_rng_state();loader=l.loader_generator.get_state();synth=l.synth_rng.clone()
    compact=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert set(compact['network'])==HEADS and ckpt.stat().st_size<2*1024**2
    # Continue uninterrupted, then repeat precisely from the saved state.
    update();expected=network_hash(l._network);expected_rng=l._capture_rng_state()
    l.restore(ckpt)
    assert network_hash(l._network)==h and rng_equal(rng,l._capture_rng_state())
    assert torch.equal(loader,l.loader_generator.get_state()) and torch.equal(synth,l.synth_rng)
    update();assert network_hash(l._network)==expected and rng_equal(expected_rng,l._capture_rng_state())
    assert losses[-2]==losses[-1]
    l.assert_stationary()
    if row_fixed:
        l.assert_old_rows(l.optimizer)
        bad=copy.deepcopy(compact);bad['old_row_reference']['rows']['backbone.head.bias'][0]+=1
        try:l.expand_delta(bad)
        except AssertionError:pass
        else:raise AssertionError('Corrupt old-row reference accepted')
    bad=dict(compact,train_seed=1994)
    try:l._restore_checkpoint_state(l.expand_delta(bad))
    except AssertionError:pass
    else:raise AssertionError('Wrong seed accepted')
    bad=dict(compact,immutable_parent={**compact['immutable_parent'],'sha256':'0'*64})
    try:l.expand_delta(bad)
    except AssertionError:pass
    else:raise AssertionError('Wrong parent accepted')
    context=batch_probe(l);probe=gradient_probe(l)
    assert probe['terms']['assignment/pool_match']['old_gradient_norm']==0
    base=config.get('replay_weight',.05)
    assert l.concm_stage1_loss_weight==l.args['concm_stage1_loss_weight']==base
    weight=l._concm_stage1_effective_weight(9)
    if config.get('replay_weight_rule')=='old_current_count':
        assert weight==2.0
        l._known_classes=6;l._total_classes=8
        assert l._concm_stage1_effective_weight(9)==3.0
        l._known_classes=4;l._total_classes=6
    else:assert weight==base
    raw=probe['terms']['synthetic/raw'];weighted=probe['terms']['synthetic/weighted']
    assert raw['weight_in_total']==weight and abs(weighted['loss']-weight*raw['loss'])<1e-6
    for part in ['old','current']:
        assert abs(weighted[part+'_gradient_norm']-weight*raw[part+'_gradient_norm'])<1e-5
    head_objective='not applicable'
    if config.get('synthetic_ce_heads')=='mean_main_few_sum':
        assert not row_fixed and l.args.get('old_classifier_rows') is None
        # Check actual sampler/heads against an independent explicit CE expression.
        with isolated_rng(l):
            fm,ff,sy=l._concm_stage1_sample_memory();b=l._backbone_module();t=l._total_classes
            main=b.head(fm)[:,:t];few=b.head_few(ff)[:,:t]
            expected=(F.cross_entropy(main+few,sy)+F.cross_entropy(main,sy)+F.cross_entropy(few,sy))/3
        with isolated_rng(l):actual=l._concm_stage1_loss()
        assert torch.equal(actual,expected)
        pars=[b.head.weight,b.head.bias,b.head_few.weight,b.head_few.bias]
        ga=torch.autograd.grad(actual,pars);ge=torch.autograd.grad(expected,pars)
        assert all(torch.equal(x,y) and torch.count_nonzero(x[6:])==0 for x,y in zip(ga,ge))
        l.args['synthetic_ce_heads']='sum_only'
        with isolated_rng(l):legacy=l._concm_stage1_loss()
        assert torch.equal(legacy,F.cross_entropy(main+few,sy))
        l.args['synthetic_ce_heads']='mean_main_few_sum'
        assert abs(raw['loss']-sum(probe['terms']['synthetic/'+n]['loss'] for n in ('main_ce','few_ce','sum_ce'))/3)<1e-5
        assert 'old_synthetic_actual_both_heads' in probe['common_logit_shift']
        assert all(probe['terms']['synthetic/'+n]['weight_in_total']==weight/3 for n in ('main_ce','few_ce','sum_ce'))
        head_objective='PASS: actual loss/gradients equal explicit CE mean; legacy default exact; probe and future mask correct'
    result={'status':'PASS','one_real_training_update':'PASS','only_four_head_tensors_have_gradients':True,
            'all_non_head_parameters_and_buffers_bitwise_unchanged':True,'future_ce_gradients_zero':True,
            'compact_restore_and_next_update_bitwise_equal':True,'normal_restore_seed_guard':'PASS','delta_parent_guard':'PASS',
            'old_rows_exact_after_adamw_decay_and_nonzero_moments':row_fixed,
            'old_row_reference_restore_guard':'PASS' if row_fixed else 'not applicable',
            'current_classifier_rows_update':True,'checkpoint_bytes':ckpt.stat().st_size,'one_batch_seconds':seconds,'peak_allocated_bytes':peak,
            'synthetic_head_objective':head_objective,'observed_losses':losses,'constant_feature_objectives_have_zero_head_gradient':True,
            'base_replay_weight':base,'actual_s1_replay_weight':weight,'probe_matches_actual_replay_weight':'PASS',
            'batch_context_and_label_independence':context,'test_predictions':0}
    write_json(out/'ENGINEERING.json',result);return result
