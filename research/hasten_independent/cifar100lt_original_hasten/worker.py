"""Official learner with the frozen APART long-tail input and endpoint recording."""
import os,sys,json,time,random,logging,traceback
from pathlib import Path
root=Path('/tmp/q1101'); official=root/'official'
os.chdir(official);sys.path.insert(0,str(official))
def write(name,value):
    p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
try:
    import numpy as np
    import torch,torchvision,open_clip
    torch.set_num_threads(4)
    from utils import data as data_module
    from utils.data_manager import DataManager
    from utils.create_hier import prepare_hierarchies
    from trainer import _set_random
    import models.hasten as method
    method.num_workers=2
    args=json.loads((root/'RUN_CONFIG.json').read_text());spec=json.loads((root/'SPLIT_SPEC.json').read_text())
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
    # Local asset binding preserves the exact pretrained model architecture/weights.
    original_create=open_clip.create_model_and_transforms
    def local_create(name,*a,**kw):
        assert name=='ViT-B-16' and kw['pretrained']=='laion400m_e32'
        kw['pretrained']=str(root/'assets/vit_b_16-laion400m_e32-55e67d44.pt')
        kw['cache_dir']=str(root/'assets/cache')
        return original_create(name,*a,**kw)
    open_clip.create_model_and_transforms=local_create
    data_module.DATA_ROOT_PATH=str(root/'assets/data')
    _set_random(1993)
    dm=DataManager('cifar224',True,1993,50,10)
    assert dm._class_order==spec['class_order'] and dm._increments==spec['increments']
    # APART resets NumPy to seed1993 for the permutation, then shuffles counts.
    counts=[int(500*(.01**(i/99))) for i in range(100)];np.random.shuffle(counts)
    assert counts==spec['train_counts_in_incremental_order']
    original_targets=dm._train_targets.copy()
    selected=np.concatenate([np.flatnonzero(original_targets==c)[:n] for c,n in enumerate(counts)])
    dm._train_data=dm._train_data[selected];dm._train_targets=original_targets[selected]
    assert len(selected)==10847 and len(np.unique(selected))==10847
    assert np.bincount(dm._train_targets,minlength=100).tolist()==counts
    assert np.bincount(dm._test_targets,minlength=100).tolist()==[100]*100
    args['hierarchy_dir']=str(prepare_hierarchies(dm,args,check_only=True))
    # No best-epoch choice: the original per-epoch diagnostic calls remain intact.
    write('SPLIT_RECEIPT.json',dict(class_order=dm._class_order,counts=counts,train_original_row_indices=selected.tolist(),test_rows=10000,trees_checked=True))
    write('ENVIRONMENT.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,open_clip=open_clip.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),cpu_affinity=sorted(os.sched_getaffinity(0))))
    args['device']=[torch.device('cuda:0')]
    model=method.Learner(args)
    assert all(not p.requires_grad for p in model._network.model.parameters())
    steps=0;last=0;stage=-1
    original_step=torch.optim.AdamW.step
    def recorded_step(opt,*a,**kw):
        global steps,last
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is not None:assert torch.isfinite(p.grad).all(),'NONFINITE_GRADIENT'
        result=original_step(opt,*a,**kw);steps+=1
        assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
        if time.time()-last>10:
            write('STATUS.json',dict(time=time.time(),phase='TRAIN',stage=stage,optimizer_steps=steps));last=time.time()
        return result
    torch.optim.AdamW.step=recorded_step
    curves=[];records=[]
    for stage in range(6):
        write('STATUS.json',dict(time=time.time(),phase='STAGE_START',stage=stage,optimizer_steps=steps))
        before=steps
        model.incremental_train(dm)
        # The same official endpoint evaluator, called once to save raw decisions.
        predictions,truth=model._eval_cnn(model.test_loader)
        pred=predictions[0][:,0];truth=np.asarray(truth)
        seen=50+10*stage
        assert len(truth)==seen*100 and np.bincount(truth,minlength=seen).tolist()==[100]*seen
        correct=[int(((pred==truth)&(truth==c)).sum()) for c in range(seen)]
        accuracy=100*int((pred==truth).sum())/len(truth);curves.append(accuracy)
        groups={}
        for name,mask in [('many',np.array(counts[:seen])>100),('medium',(np.array(counts[:seen])>=20)&(np.array(counts[:seen])<=100)),('few',np.array(counts[:seen])<20)]:
            ids=np.flatnonzero(mask);groups[name]=dict(correct=sum(correct[c] for c in ids),total=100*len(ids))
        np.savez(root/f'predictions_stage{stage}.npz',prediction=pred,truth=truth)
        record=dict(stage=stage,seen=seen,accuracy=accuracy,correct_per_class=correct,total_per_class=[100]*seen,frequency_groups=groups,optimizer_steps=steps-before,cumulative_optimizer_steps=steps)
        records.append(record);write('STAGE_RESULTS.json',records)
        # Frozen backbone lives once in assets; retain only learned state/statistics.
        state={k:v.detach().cpu() for k,v in model._network.state_dict().items() if not k.startswith(('model.','visual.')) and k!='visual_proj'}
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
        hierarchy={k:v.detach().cpu() for k,v in model.hierarchy_manager.state_dict().items() if not k.startswith('network.')}
        torch.save(dict(network=state,prototype=[x.cpu() for x in model.prototype],cov=[x.cpu() for x in model.cov],hierarchy=hierarchy,args={k:str(v) if k=='device' else v for k,v in args.items()}),root/f'head_stage{stage}.pt')
        model.after_task()
    ref=spec['APART_reference_curve']
    result=dict(method='official_HASTEN_APART_LT_input',seed=1993,curve=curves,average=sum(curves)/6,final=curves[-1],APART_historical_curve=ref,APART_average=sum(ref)/6,APART_final=ref[-1],average_delta_pp=sum(curves)/6-sum(ref)/6,final_delta_pp=curves[-1]-ref[-1],optimizer_steps=steps,backbones_matched=False,complete=True)
    write('RESULTS.json',result);write('STATUS.json',dict(time=time.time(),phase='COMPLETE',optimizer_steps=steps))
except BaseException:
    write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
