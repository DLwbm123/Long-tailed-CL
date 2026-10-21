"""One frozen post-Task1 NIH control using the original sealed source tree."""
import os,sys,time,json,random
from pathlib import Path
import numpy as np
import torch
from common import settings,write,report


def main():
    cfg,out=settings();out.mkdir(exist_ok=False);start=time.time();counts=dict(qualification_training_encodes=0,scientific_training_encodes=0,validation_encodes=0,encoder_optimizer_updates=0,analytic_solves=0)
    sys.path.insert(0,str(Path(cfg['source'])/'tools'))
    legacy=Path(json.loads(Path(cfg['source_input']).read_text())['legacy_repo'])
    sys.path.insert(1,str(legacy/'third_party/APART'))
    import prototype_coherent as method
    import module_additions as modules
    import core1_competition as competition
    import edge_competition as edge
    from run_prototype_coherent import fit_split,cpu_bank
    from run_prototype_single import Images,manifests
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from torch.utils.data import DataLoader
    torch.set_num_threads(4);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    def status(phase,**extra):write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),**counts,**extra))
    def budget():
        assert time.time()<cfg['hard_deadline_epoch']-60,'Original deadline'
        assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is True,'Original precision settings changed'
        assert torch.cuda.memory_reserved()<cfg['GPU_reserved_budget_bytes'],'GPU reserved hard limit'
    def load(rows,task):
        return DataLoader(Images(rows,cfg['images'],transform(False),cfg['seed']+task*100003),batch_size=64,num_workers=cfg['workers'],shuffle=False,multiprocessing_context='spawn' if cfg['workers'] else None)
    def encode(rows,task,field):
        xs=[];ys=[]
        for images,labels in load(rows,task):
            budget()
            with torch.no_grad():z=encoder(images.cuda()).detach().cpu().numpy()
            counts[field]+=len(z);xs.append(z);ys.append(labels.numpy());status('encoding',task=task)
        return np.concatenate(xs),np.concatenate(ys)
    try:
        from verify import self_check
        math_error=self_check()
        status('source_qualification',synthetic_dense_error=math_error)
        original=json.loads(Path(cfg['source_input']).read_text())
        for k in ['source_commit','seed','order','task_sizes']:assert original[k]==cfg[k],k
        assert original['name']=='NIHCXRLT' and original['module']=='base' and original['method']=='static_pc'
        train,val=manifests(dict(original,manifest=cfg['manifest']))
        assert len(train)==68058 and len(val)==300
        source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
        assert source['seen']==cfg['order'][:10] and source['steps']==1504 and source['module']=='base'
        assert tuple(source['head'].shape)==(1536,10) and bool((source['selected_coefficients']==0).all())
        encoder=ApartFeatures(original['legacy_repo'],original['weight'],20,'cuda:0',cfg['seed'])
        encoder.load_state_dict(source['model'],strict=True);encoder.requires_grad_(False).eval()
        budget()
        write(out/'RUNTIME_PRECISION.json',dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,float32_matmul_precision=torch.get_float32_matmul_precision(),torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),original_source_precision_inherited=True))
        parameters=dict(encoder.named_parameters())
        assert source['adapter'] and all(torch.equal(parameters[k].detach().cpu(),v) for k,v in source['adapter'].items())
        bank={k:(v.cuda() if torch.is_tensor(v) else [dict(c,**{a:(b.cuda() if torch.is_tensor(b) else b) for a,b in c.items()}) for c in v] if k=='components' else list(v)) for k,v in source['bank'].items()}
        assert bank['mu'].shape==(10,1536) and bank['Q'].shape==(10,1536,1536)
        assert bank['n']==[sum(r['label']==c for r in train) for c in cfg['order'][:10]]
        for k in range(10):
            comp=[c for c in bank['components'] if c['label']==k]
            assert sum(c['count'] for c in comp)==bank['n'][k] and abs(float(sum(c['mass'] for c in comp))-1)<1e-8
        native,R,_=modules.head(bank,'base','NIHCXRLT',source['seen']);counts['analytic_solves']+=1
        base=competition.competition(bank,native)
        zero=torch.zeros(8,dtype=torch.float64,device='cuda');features=edge.descriptors(bank,native,0)
        assert torch.equal(edge.allocation(base,features,zero),base)
        rebuilt,audit=competition.bank_head(bank,source['boundary_pair_weights'].cuda(),regularizer=R);counts['analytic_solves']+=1
        error=float((rebuilt.cpu()-source['head'].double()).norm()/source['head'].double().norm())
        assert error<=2e-6 and audit['relative_residual']<=1e-8
        qrows=[r for r in train if r['label'] in cfg['order'][10:12]][:32];assert len(qrows)==32
        q1,_=encode(qrows,2,'qualification_training_encodes');q2,_=encode(qrows,2,'qualification_training_encodes')
        assert np.array_equal(q1,q2) and abs(np.linalg.norm(q1,axis=1)-1).max()<=1e-5
        write(out/'QUALIFICATION.json',dict(status='PASS',source_head_relative_error=error,source_head_solve=audit,real_repeat_max_error=float(abs(q1-q2).max()),source_steps=1504,train_encodes=counts['qualification_training_encodes'],validation_encodes=0,static_zero_allocation_equal=True))
        heads=[source['head'].double().numpy()];records=[dict(task=1,seen=source['seen'],inherited=True,source_head_relative_error=error)];seen=source['seen'].copy()
        for task in range(2,7):
            classes=cfg['order'][10+2*(task-2):12+2*(task-2)];rows=[r for r in train if r['label'] in classes]
            x,y_original=encode(rows,task,'scientific_training_encodes');old_count=len(seen);seen+=classes
            x=torch.as_tensor(x,device='cuda',dtype=torch.float64);y=torch.as_tensor([seen.index(int(c)) for c in y_original],device='cuda')
            fit,_=fit_split(rows,original['split_seed']);fit=torch.as_tensor(fit,device='cuda');seeds=method.seed_components(x[fit],y[fit]);group,difficulty=method.memberships(x,y,seeds)
            weights=method.sample_weights(y,group,difficulty,x.new_zeros(len(seeds)))
            assert all(torch.allclose(weights[y==j],weights.new_full((int((y==j).sum()),),1/int((y==j).sum())),atol=1e-14,rtol=1e-12) for j in range(old_count,len(seen)))
            previous=bank;bank=method.append(bank,x,y,group,weights)
            assert torch.equal(bank['mu'][:old_count],previous['mu']) and torch.equal(bank['Q'][:old_count],previous['Q'])
            status('fitting',task=task)
            native,R,activation=modules.head(bank,'base','NIHCXRLT',seen);counts['analytic_solves']+=1
            pairs=competition.competition(bank,native)
            head,audit=competition.bank_head(bank,pairs,regularizer=R);counts['analytic_solves']+=1
            assert audit['relative_residual']<=1e-8
            heads.append(head.cpu().numpy());records.append(dict(task=task,seen=seen.copy(),n=bank['n'].copy(),solve=audit,extra_columns=activation['extra_columns']))
            tmp=out/'bank.private.tmp.pt';torch.save(cpu_bank(bank),tmp);tmp.replace(out/'bank.private.pt')
            del previous,x,y
        assert counts['scientific_training_encodes']==7929
        assert all(torch.equal(v.detach().cpu(),source['model'][k]) for k,v in encoder.state_dict().items()),'Frozen model changed'
        for k in ['mu','Q']:assert torch.equal(bank[k][:10].cpu(),source['bank'][k]),'Inherited bank changed'
        np.savez(out/'heads.private.npz',**{f'stage{j+1}':w for j,w in enumerate(heads)})
        write(out/'FIT_SEAL.json',dict(status='SEALED',stages=records,encoder_unchanged=True,old_bank_unchanged=True,**counts))
        del source,bank,R,native,head;torch.cuda.empty_cache();budget()
        x,labels=encode(val,6,'validation_encodes');assert len(x)==300
        np.savez(out/'query.private.npz',x=x,labels=labels)
        results=report(x,labels,heads,cfg['order'])
        historical=json.loads(Path(cfg['historical_metrics']).read_text())
        recalls=historical['stages'][0]['per_class_recall']
        expected={k:round(float(v)*15) for k,v in recalls.items()}
        write(out/'TASK1_PARITY.json',dict(expected=expected,actual=results['stages'][0]['correct']))
        assert results['stages'][0]['correct']==expected,'Task1 validation parity failed'
        write(out/'RESULTS.json',results)
        no_loss=results['final_correct']>=74 and results['mean_BA_percent']>=28.319223985890655 and results['tail4_correct']>=15
        improved=no_loss and (results['final_correct']>=77 or results['mean_BA_percent']>=29.319223985890655)
        write(out/'DECISION.json',dict(no_loss_gate=bool(no_loss),improvement_gate=bool(improved),scope='single_NIH_A_frozen_control_development',no_parameter_search=True))
        write(out/'status.private.json',dict(state='COMPLETE_PENDING_VERIFICATION',updated=time.time(),**counts))
    except BaseException as e:
        write(out/'status.private.json',dict(state='FAILED',error=str(e),updated=time.time(),**counts));raise
    finally:
        write(out/'COSTS.json',dict(worker_seconds=time.time()-start,**counts,inherited_task1_adapter_updates=1504,source_experiment_cost_not_reset=True,CPU_threads=4,GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__':main()
