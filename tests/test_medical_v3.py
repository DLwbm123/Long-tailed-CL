"""Run against the locked remote parents; actual legacy/new training paths, disposable updates."""
import copy
import importlib.util
import json
import time
from pathlib import Path
import torch
from run_medical_v2 import args_for,MedicalLearner,network_hash,sha,write_json
from diagnose_medical_v3 import load_parent,fixed_probe,gradient_probe,batch_probe,isolated_rng,dataset,loader
from utils.medical_v2 import effective_optimizer

def engineering(config):
    out=Path(config['output'])/'engineering';out.mkdir(exist_ok=True,parents=True)
    lock=json.loads((Path(config['v2_output'])/'ALL_CHECKPOINTS_LOCK.json').read_text())
    e=next(x for x in lock['checkpoints'] if (x['seed'],x['branch'],x['session'])==(1993,'B',0))
    l,_=load_parent(config,e,out/'scratch')
    legacy_path=Path(config['v2_runtime']).parent/'code/third_party/APART/models/apart.py'
    v2=json.loads(Path(config['v2_runtime']).read_text())
    assert sha(legacy_path)==v2['code_sha256']['third_party/APART/models/apart.py']
    spec=importlib.util.spec_from_file_location('n3_legacy',legacy_path);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    hashes=[];losses=[];gradients=[];raw_old=[]
    original_args=copy.deepcopy(l.args)
    for scope in ('legacy','current','all_seen'):
        l.args=copy.deepcopy(original_args);l.restore(e['path'])
        l._cur_task=1;l._known_classes=4;l._total_classes=6
        x,y=fixed_probe(l);x=x[:4];y=y[:4]
        l.args=dict(l.args,tuned_epoch=1)
        if scope!='legacy':l.args['real_ce_scope']=scope
        l._v2_input_hook=lambda *a:None
        l._v2_epoch_hook=lambda epoch,opt,sched,stats:losses.append(stats['loss'])
        def capture(epoch,batch,loss):
            g={n:p.grad.detach().cpu().clone() for n,p in l._network.backbone.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(v).all() for v in g.values())
            gradients.append(g);raw_old.append(float(g['head.weight'][:4].norm()))
        l._v2_gradient_hook=capture;l._v2_start_epoch=0
        optimizer=effective_optimizer(l._network.backbone)
        fn=legacy.Learner._init_train if scope=='legacy' else type(l)._init_train
        fn(l,[(torch.arange(len(x)),x,y)],None,optimizer,None)
        hashes.append(network_hash(l._network))
    assert losses[0]==losses[1] and hashes[0]==hashes[1],'BLOCKED_DEFAULT_REGRESSION'
    assert gradients[0].keys()==gradients[1].keys()
    assert all(torch.equal(gradients[0][k],gradients[1][k]) for k in gradients[0]),'BLOCKED_DEFAULT_GRADIENT'
    assert raw_old[0]==raw_old[1]==0 and raw_old[2]>0,'BLOCKED_OLD_ROW_GRADIENT'
    assert all((g['head.weight'][6:]==0).all() and (g['head_few.weight'][6:]==0).all() for g in gradients),'BLOCKED_FUTURE_GRADIENT'
    # Loss invalid candidate value must fail immediately.
    l.args['real_ce_scope']='future';failed=False
    try:l._init_train([],None,optimizer,None)
    except ValueError:failed=True
    assert failed
    # Actual C parent memory and each real component are independently checked.
    c=next(x for x in lock['checkpoints'] if (x['seed'],x['branch'],x['session'])==(1993,'C',0))
    cl,state=load_parent(config,c,out/'scratch_c');cl._cur_task=1;cl._known_classes=4;cl._total_classes=6
    probe=gradient_probe(cl)
    for name in ('main_ce','sum_ce','few_pool_ce'):
        assert probe['terms']['all_seen/'+name]['old_gradient_norm']>0
        assert probe['terms']['current/'+name]['old_gradient_norm']==0
    context=batch_probe(cl)
    report={'status':'ENGINEERING_PASS','default_loss_values':losses[:2],'default_network_hashes':hashes[:2],
            'default_loss_network_gradient_bitwise_equal':True,'old_head_gradient_norm':raw_old,
            'future_head_gradients_zero':True,'invalid_scope_rejected':True,'label_blind_inference':'PASS',
            'gradient_probe':probe,'batch_context':context,'test_predictions':0}
    write_json(out/'ENGINEERING.json',report)
    return report

def preflight_budget(config):
    out=Path(config['output']);lock=json.loads((Path(config['v2_output'])/'ALL_CHECKPOINTS_LOCK.json').read_text())
    l,_=load_parent(config,lock['checkpoints'][0],out/'engineering/throughput')
    d=dataset(l.config,l.order,4,'train');d.rows=d.rows[:144]
    torch.cuda.reset_peak_memory_stats();times=[]
    with isolated_rng(l),torch.no_grad():
        l._network.eval()
        for _,x,y in loader(d):
            x=x.cuda();torch.cuda.synchronize();t=time.monotonic();l._network(x);torch.cuda.synchronize();times.append(time.monotonic()-t)
    rate=max(times[1:])/48
    counts=[10529,3263,2835,1306,458,256,44,27]
    p0images=2*sum(sum(counts[c] for c in order[:seen]) for order in config['class_orders'].values() for seen in (4,6,8))+18*295
    history=[]
    for p in Path(config['v2_output']).glob('199?_[BC]/TRAJECTORY_COMPLETE.json'):
        for r in json.loads(p.read_text())['sessions']:
            if r['session']>0:history.append(r['epoch_seconds']/r['train_images'])
    increment=sum(sum(counts[c] for c in order[4:]) for order in config['class_orders'].values())
    estimate=1.5*(rate*p0images+rate*2*(18718+295)+max(history)*2*10*increment+120*8+increment*rate)
    r={'status':'PASS' if estimate<8*3600 else 'BLOCKED_BUDGET','inference_seconds_per_image':rate,'observed_batch_seconds':times,
       'historical_same_host_training_seconds_per_image_upper':max(history),'p0_forward_image_estimate':p0images,
       'estimated_gpu_process_seconds_with_50pct_reserve':estimate,'budget_seconds':8*3600,
       'p2_full_epoch_gate_pending':True,'inference_peak_bytes':torch.cuda.max_memory_allocated(),'test_predictions':0}
    write_json(out/'engineering/PREFLIGHT_BUDGET.json',r)
    assert r['status']=='PASS',str(r)
    return r
