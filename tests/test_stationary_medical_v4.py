"""One real update, exact frozen-state and resume/continuation checks."""
import copy
import time
from pathlib import Path
import torch
from run_medical_v2 import write_json,network_hash,rng_equal
from stationary_medical_v4 import fork,HEADS
from diagnose_medical_v3 import fixed_probe,batch_probe,gradient_probe
from utils.medical_v2 import effective_optimizer

def engineering(config):
    out=Path(config['output'])/'engineering';out.mkdir(parents=True,exist_ok=True)
    l=fork(config,1993,out/'scratch');l._network.cuda()
    original=network_hash(l._network);l.args['tuned_epoch']=1
    x,y=fixed_probe(l);x=x[:4];y=y[:4]
    batch=[(torch.arange(len(x)),x,y)]
    losses=[];grad={}
    def capture(epoch,batch,loss):
        for name,p in l._network.named_parameters():
            if p.grad is not None:grad[name]=p.grad.detach().cpu().clone()
        assert set(grad)==HEADS and all(torch.isfinite(g).all() for g in grad.values())
    l._v2_gradient_hook=capture
    l._v2_epoch_hook=lambda epoch,opt,sched,stats:losses.append(stats['loss'])
    def update():
        l._v2_start_epoch=0
        l._init_train(batch,None,l.optimizer,None)
    l.optimizer=effective_optimizer(l._network.backbone)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();update();torch.cuda.synchronize()
    seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()
    l.assert_stationary();assert network_hash(l._network)!=original
    assert all((g[6:]==0).all() for g in grad.values())
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
    assert losses[1]==losses[2]
    l.assert_stationary()
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
    result={'status':'PASS','one_real_training_update':'PASS','only_four_head_tensors_have_gradients':True,
            'all_non_head_parameters_and_buffers_bitwise_unchanged':True,'future_ce_gradients_zero':True,
            'compact_restore_and_next_update_bitwise_equal':True,'normal_restore_seed_guard':'PASS','delta_parent_guard':'PASS',
            'checkpoint_bytes':ckpt.stat().st_size,'one_batch_seconds':seconds,'peak_allocated_bytes':peak,
            'observed_losses':losses,'constant_feature_objectives_have_zero_head_gradient':True,
            'base_replay_weight':base,'actual_s1_replay_weight':weight,'probe_matches_actual_replay_weight':'PASS',
            'batch_context_and_label_independence':context,'test_predictions':0}
    write_json(out/'ENGINEERING.json',result);return result
