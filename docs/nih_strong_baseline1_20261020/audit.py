"""Complete only retained ImageNet checkpoint verification after user stopped CLIP."""
import os, json, time, gc, resource
from pathlib import Path
import numpy as np
import torch
from common import rows, Images, Model, write

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
root=Path(cfg['output']); out=Path(cfg['audit_output']); start=time.time()
torch.set_num_threads(cfg['CPU_threads']); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
assert cfg['original_start_epoch']==1792473428.1970181
assert cfg['hard_deadline_epoch']==1792559828.1970181
assert not (out/'VERIFICATION.json').exists()
data, audit=rows(cfg); y=np.array([z[1] for z in data['val']])
curves=json.loads((root/'LEARNING_CURVES.json').read_text())
results=json.loads((root/'RESULTS.json').read_text())
checked=[]; ties=[]; encoded=0; maxdiff=0.
for arm in ['imagenet_frozen_LA','imagenet_full_LA']:
    history=[h for h in curves if h['arm']==arm]
    assert [h['epoch'] for h in history]==list(range(1,31))
    item=next(z for z in results if z['arm']==arm)
    assert item['updates']==15960
    chosen=item['best_epoch']; earliest=max(history,key=lambda z:z['metrics']['correct'])['epoch']
    ties.append(dict(arm=arm,saved_best_epoch=chosen,earliest_max_correct_epoch=earliest,
                     integer_tie_selection_deviation=chosen!=earliest))
    for tag, expected in [('epoch10',10),('best',chosen),('last',30)]:
        assert time.time()<cfg['audit_deadline_epoch']-30
        cp=torch.load(root/arm/(tag+'.private.pt'),map_location='cpu',weights_only=False)
        assert cp['epoch']==expected and cp['arm']==arm
        model=Model(cfg,'imagenet');model.load_state_dict(cp['model'],strict=True)
        del cp;model.cuda().eval();scores=np.empty((300,20),dtype=np.float32)
        loader=torch.utils.data.DataLoader(Images(data['val'],'imagenet',False,None),batch_size=cfg['micro_batch_size'],shuffle=False,num_workers=0)
        with torch.inference_mode():
            for x,_,ids in loader:
                assert time.time()<cfg['audit_deadline_epoch']-30
                with torch.autocast('cuda',dtype=torch.bfloat16):s=model(x.cuda())
                scores[ids.numpy()]=s.float().cpu().numpy();encoded+=len(ids)
                assert encoded<=1800
                write(out/'status.private.json',dict(state='RUNNING',arm=arm,checkpoint=tag,validation_encodes=encoded,updated=time.time()))
        saved=np.load(root/arm/f'val_epoch_{expected}.private.npz')
        diff=float(np.abs(scores-saved['scores']).max());maxdiff=max(maxdiff,diff)
        assert diff<1e-4 and np.array_equal(scores.argmax(1),saved['scores'].argmax(1))
        cm=np.bincount(20*y+scores.argmax(1),minlength=400).reshape(20,20)
        stored=history[expected-1]['metrics'];rec=np.diag(cm)/cm.sum(1)
        assert np.all(cm.sum(1)==15) and np.array_equal(cm,np.array(stored['confusion']))
        assert np.max(np.abs(rec-stored['recall']))<1e-12
        assert abs(float(rec.mean())-stored['BA'])<1e-12
        assert int(np.diag(cm).sum())==stored['correct']
        checked.append(dict(arm=arm,checkpoint=tag,epoch=expected,BA=float(rec.mean()),correct=int(np.diag(cm).sum())))
        del model,scores,loader;gc.collect();torch.cuda.empty_cache()
assert encoded==1800
write(out/'VERIFICATION.json',dict(state='PASS_IMAGENET_ONLY',checkpoints=checked,validation_image_encodes=encoded,training_image_encodes=0,optimizer_updates=0,max_score_error=maxdiff,patient_overlap=0,checkpoint_selection=ties,all_four_arms_verified=False,CLIP_restarted_or_rescored=False,pack_recreated=False,verification_seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch']))
write(out/'status.private.json',dict(state='COMPLETE_IMAGENET_ONLY',validation_encodes=encoded,updated=time.time()))
