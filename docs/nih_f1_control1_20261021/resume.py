"""Recover only the pre-query seal write; reuse all completed training encodes."""
import json,sys,time,random
from pathlib import Path
import numpy as np
import torch
from common import settings,write,report


def main():
    cfg,out=settings();start=time.time();cost=json.loads((out/'COSTS.json').read_text());failed=json.loads((out/'status.private.json').read_text())
    assert failed['error']=="dict() got multiple values for keyword argument 'validation_encodes'"
    assert failed['scientific_training_encodes']==7929 and failed['qualification_training_encodes']==64 and failed['validation_encodes']==0 and failed['analytic_solves']==12
    assert not (out/'query.private.npz').exists() and not (out/'FIT_SEAL.json').exists()
    assert json.loads((out/'QUALIFICATION.json').read_text())['status']=='PASS'
    sys.path.insert(0,str(Path(cfg['source'])/'tools'))
    original=json.loads(Path(cfg['source_input']).read_text());sys.path.insert(1,str(Path(original['legacy_repo'])/'third_party/APART'))
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from run_prototype_single import Images,manifests
    from torch.utils.data import DataLoader
    torch.set_num_threads(4);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    counts={k:cost[k] for k in ['qualification_training_encodes','scientific_training_encodes','validation_encodes','encoder_optimizer_updates','analytic_solves']}
    try:
        source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
        bank=torch.load(out/'bank.private.pt',map_location='cpu',weights_only=False)
        heads_file=np.load(out/'heads.private.npz');heads=[heads_file[f'stage{j}'] for j in range(1,7)]
        assert source['steps']==1504 and source['seen']==cfg['order'][:10] and source['module']=='base'
        assert np.array_equal(heads[0],source['head'].double().numpy())
        assert all(w.shape==(1536,k) and np.isfinite(w).all() for w,k in zip(heads,[10,12,14,16,18,20]))
        for k in ['mu','Q']:assert torch.equal(bank[k][:10],source['bank'][k])
        train,val=manifests(original)
        assert bank['n']==[sum(r['label']==c for r in train) for c in cfg['order']]
        assert len(val)==300
        # The original model/bank assertions precede np.savez and the failed dict construction.
        terminal=json.loads((out.parent/'attempt4'/'run_terminal.private.json').read_text())
        assert (out/'heads.private.npz').stat().st_mtime<=terminal['finished']
        seal=dict(status='SEALED_RECOVERED_BEFORE_QUERY',stages=[dict(task=j,seen=cfg['order'][:w.shape[1]],n=bank['n'][:w.shape[1]]) for j,w in enumerate(heads,1)],encoder_unchanged=True,old_bank_unchanged=True,model_assertion_evidence='completed original source assertion before heads save and failed seal write',recovery_reencodes=0,**counts)
        write(out/'FIT_SEAL.json',seal)
        encoder=ApartFeatures(original['legacy_repo'],original['weight'],20,'cuda:0',cfg['seed']);encoder.load_state_dict(source['model'],strict=True);encoder.requires_grad_(False).eval()
        del source,bank
        loader=DataLoader(Images(val,cfg['images'],transform(False),cfg['seed']+6*100003),batch_size=64,num_workers=cfg['workers'],shuffle=False,multiprocessing_context='spawn')
        xs=[];ys=[]
        for images,labels in loader:
            assert time.time()<cfg['hard_deadline_epoch']-60 and torch.cuda.memory_reserved()<cfg['GPU_reserved_budget_bytes']
            with torch.no_grad():x=encoder(images.cuda()).cpu().numpy()
            xs.append(x);ys.append(labels.numpy());counts['validation_encodes']+=len(x)
            write(out/'status.private.json',dict(state='RUNNING',phase='recovered_validation',updated=time.time(),**counts))
        x=np.concatenate(xs);labels=np.concatenate(ys);assert len(x)==300
        np.savez(out/'query.private.npz',x=x,labels=labels)
        results=report(x,labels,heads,cfg['order']);hist=json.loads(Path(cfg['historical_metrics']).read_text());expected={k:round(float(v)*15) for k,v in hist['stages'][0]['per_class_recall'].items()}
        write(out/'TASK1_PARITY.json',dict(expected=expected,actual=results['stages'][0]['correct']));assert results['stages'][0]['correct']==expected,'Task1 validation parity failed'
        write(out/'RESULTS.json',results)
        no_loss=results['final_correct']>=74 and results['mean_BA_percent']>=28.319223985890655 and results['tail4_correct']>=15
        improved=no_loss and (results['final_correct']>=77 or results['mean_BA_percent']>=29.319223985890655)
        write(out/'DECISION.json',dict(no_loss_gate=bool(no_loss),improvement_gate=bool(improved),scope='single_NIH_A_frozen_control_development',no_parameter_search=True))
        write(out/'status.private.json',dict(state='COMPLETE_PENDING_VERIFICATION',updated=time.time(),**counts))
    except BaseException as e:
        write(out/'status.private.json',dict(state='FAILED_RECOVERY',error=str(e),updated=time.time(),**counts));raise
    finally:
        cost.update(counts);cost.update(recovery_worker_seconds=time.time()-start,recovery_train_encodes=0,main_attempts_failed=4,recovery_GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved());write(out/'COSTS.json',cost)

if __name__=='__main__':main()
