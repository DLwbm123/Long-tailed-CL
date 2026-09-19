"""Fixed beta=1 Task4 fork; inherited scientific dependencies stay immutable."""
import os,time,gc,copy,functools,sys,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_ct6f as w
import run_ct3p as base
from run_ct3p import FDStudent,read,sha,write_json,jsonl,npz,network_hash,joint,ridge
from ct3p_core import fd
from run_medical_v2 import rng_equal,WEIGHT_SHA
from resume_ct3p import finish_task
from resume_ct6f import exchange

def coefficient_check():
    x=torch.tensor([.3,-.8],dtype=torch.float64,requires_grad=True)
    real=(x-1).square().sum();dist=(x+.2).square().mean()
    g0=torch.autograd.grad(real,x,retain_graph=True)[0]
    gd=torch.autograd.grad(dist,x,retain_graph=True)[0]
    for beta in (1.,10.):
        g=torch.autograd.grad(real+beta*dist,x,retain_graph=True)[0]
        assert torch.allclose(g,g0+beta*gd,atol=1e-12,rtol=1e-12)
    return dict(status='PASS',beta10_regression=True,beta1_exact_gradient_difference=True)

class Student(FDStudent):
    def __init__(self,*args):
        super().__init__(*args);self.beta=1.
    def _ct3p_add_loss(self,real,x,epoch,batch):
        if self.beta==0:return real
        loss,dist=fd(self._network,self.teacher,x,self)
        self.r.count(self.r.phase+'_wrapper_calls',2);self.r.count(self.r.phase+'_internal_encoder_calls',6)
        self.fdparts.append(dict(FD=float(loss.detach()),FD_p95=float(torch.quantile(dist,.95)),batch=len(x)))
        if batch==0:
            names=[(k,v) for k,v in self._network.named_parameters() if v.requires_grad and ('.pool.pool.' in k or '.pool_few.pool.' in k)]
            g0=torch.autograd.grad(real,[v for _,v in names],retain_graph=True,allow_unused=True)
            g1=torch.autograd.grad(loss,[v for _,v in names],retain_graph=True,allow_unused=True)
            rows={}
            for group,part in [('main','.pool.pool.'),('few','.pool_few.pool.')]:
                ix=[i for i,(k,_) in enumerate(names) if part in k];n0=sum(float(g0[i].detach().square().sum()) for i in ix if g0[i] is not None)**.5;n1=sum(float(g1[i].detach().square().sum()) for i in ix if g1[i] is not None)**.5
                dot=sum(float((g0[i].detach()*g1[i].detach()).sum()) for i in ix if g0[i] is not None and g1[i] is not None)
                rows[group]=dict(real_norm=n0,FD_norm=n1,weighted_ratio=None if n0==0 else self.beta*n1/n0,cosine=None if n0*n1==0 else dot/(n0*n1))
            jsonl(self.r.pub/('FEATURE_DISTILLATION_AUDIT.jsonl' if self.r.phase=='formal' else 'engineering_FD.jsonl'),dict(dataset=self.name,seed=self.seed,task=self.task+1,epoch=epoch+1,groups=rows))
        return real+self.beta*loss
    def save_new(self,path,resume):
        p=dict(delta={k:v.detach().cpu().clone() for k,v in self._network.state_dict().items() if k in self.nonshared},nonshared_keys=sorted(self.nonshared),
            network_sha256=network_hash(self._network),shared_sha256=self.shared_hash,weight_sha256=WEIGHT_SHA,code_sha256=self.r.code,
            dataset=self.name,seed=self.seed,stream='F',task=self.task,epoch=self.current_epoch,known=self._known_classes,seen=self._total_classes,order=self.order,
            stats=self.stats,T=self.T,optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state(),
            synthesis_rng=self.synth.get_state(),steps=self.steps,records=self.records,teacher_source=self.teacher_source,teacher_hash=self.teacher_hash,
            manifest_sha256=self.r.lock[self.name]['manifest_sha256'],args=dict(self.args,device=['cuda:0']),beta=1.,intervention_source=self.r.cfg['source_commit'],intervention_protocol=sha(self.r.pub/'PROTOCOL_LOCK.json'))
        if resume:p.update(anchor=self.last_anchor,anchor_stats=self.anchor_stats,anchor_T=self.anchor_T,epoch_stats=self.epoch_stats,epoch_T=self.epoch_T,last_features=self.last_features,
            anchor_routes=self.anchor_routes,teacher_delta={k:v.detach().cpu().clone() for k,v in self.teacher.state_dict().items() if k in self.nonshared})
        temp=Path(str(path)+'.part');torch.save(p,temp);self.r.resources();temp.replace(path);return p
    def restore_new(self,path,resume):
        p=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in [('dataset',self.name),('seed',self.seed),('stream','F'),('order',self.order),('weight_sha256',WEIGHT_SHA),('code_sha256',self.r.code),('shared_sha256',self.shared_hash),('manifest_sha256',self.r.lock[self.name]['manifest_sha256']),('nonshared_keys',sorted(self.nonshared)),('beta',1.),('intervention_source',self.r.cfg['source_commit']),('intervention_protocol',sha(self.r.pub/'PROTOCOL_LOCK.json'))]:assert p[k]==v,('BLOCKED_RESTORE',k)
        assert p['args']==dict(self.args,device=['cuda:0'])
        state=self._network.state_dict();state.update(p['delta']);self._network.load_state_dict(state,strict=True);assert network_hash(self._network)==p['network_sha256']
        self.task=p['task'];self._cur_task=self.task;self.current_epoch=p['epoch'];self._known_classes=p['known'];self._total_classes=p['seen'];self.current=range(p['known'],p['seen'])
        self.stats=p['stats'];self.T=p['T'];self.steps=p['steps'];self.records=p['records'];self.teacher_source=p['teacher_source'];self.teacher_hash=p['teacher_hash']
        if self.optimizer is not None:self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler'])
        self.loader_generator.set_state(p['loader_rng']);self.synth.set_state(p['synthesis_rng']);self._restore_rng_state(p['rng'])
        if resume:
            self.anchor_routes=p['anchor_routes'];self.last_anchor=p['anchor'];self.anchor_stats=p['anchor_stats'];self.anchor_T=p['anchor_T'];self.epoch_stats=p['epoch_stats'];self.epoch_T=p['epoch_T'];self.last_features=p['last_features']
            st=self.teacher.state_dict();st.update(p['teacher_delta']);self.teacher.load_state_dict(st,strict=True);assert network_hash(self.teacher)==self.teacher_hash
            self._v2_start_epoch=p['epoch']
        return p


class Run(w.Full):
    exchange=exchange
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert self.scope==(name,seed,4), 'BLOCKED_NON_TASK4'
        return super().dataset(name,seed,classes,train,split,smoke)
    def fetch(self,e):
        self.exchange('GET',e['name'],e['sha256'])
        src=self.private/'parent.part';assert sha(src)==e['sha256']
        target=self.private/'parents'/e['name'];assert not target.exists();src.replace(target)
        return target
    def save_candidate(self,l,method,state,source):
        return super().save_candidate(l,{'C2':'B1A','C3':'B1T'}[method],state,source)
    def resources(self,enforce=True):
        x=super().resources(enforce)
        if enforce:assert self.archived<3*1024**3,'BLOCKED_ARCHIVE_BUDGET'
        return x


def fork(r,e):
    path=r.fetch(e);l=Student(r,e['dataset'],e['seed'])
    # Explicit fork boundary: ordinary parent restore retains every beta10 check.
    p=FDStudent.restore_new(l,path,False)
    assert (p['task'],p['epoch'],p['seen'],p['beta'])==(2,10,6,10.)
    assert network_hash(l._network)==e['network_sha256'] and rng_equal(l._capture_rng_state(),p['rng'])
    assert torch.equal(l.loader_generator.get_state(),p['loader_rng']) and torch.equal(l.synth.get_state(),p['synthesis_rng'])
    for bank,method in [(l.stats,'C2'),(l.T,'C3')]:
        assert bank['mu'].shape==(1536,6) and all(np.isfinite(bank[k]).all() for k in ('mu','S','v'))
        np.testing.assert_array_equal(bank['n'],[r.lock[l.name]['train_counts'][c] for c in l.order[:6]])
        with np.load(Path(r.cfg['ct3p_root'])/'output/private/banks'/f'{l.name}_{l.seed}_{method}_t03.npz') as f:
            np.testing.assert_allclose(ridge(bank)[0],f['W'],atol=1e-9,rtol=1e-9)
    l.teacher_source=e;l.beta=1.
    row=dict(dataset=l.name,seed=l.seed,parent_sha256=e['sha256'],parent_network=e['network_sha256'],
        ordinary_parent_restore=True,args_unchanged=True,network_RNG_loader_synthesis=True,A_T_W=True,
        intervention=dict(parent_beta=10.,new_beta=1.,task=4,epochs=10),phase=r.mode)
    jsonl(r.pub/'FORK_AUDIT.jsonl',row)
    path.unlink() # Verified duplicate; original parent archive remains immutable.
    return l,p


def qualify(r,parents):
    tests=coefficient_check();rows=[]
    for e in parents:
        l,p=fork(r,e);r.permit(l.name,l.seed,4);l.task_setup(3)
        assert l.teacher_hash==e['network_sha256'] and all(not v.requires_grad for v in l.teacher.parameters())
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(l._network.parameters(),l.teacher.parameters()))
        assert rng_equal(l._capture_rng_state(),p['rng']) and torch.equal(l.loader_generator.get_state(),p['loader_rng'])
        # Wrong intervention beta must fail before changing network state.
        candidate=r.private/'engineering.pt';v=l.save_new(candidate,False)
        assert v['beta']==1. and v['intervention_source']==r.cfg['source_commit']
        l.restore_new(candidate,False)
        for field,bad in [('beta',10.),('intervention_source','wrong'),('intervention_protocol','wrong')]:
            original_value=v[field];v[field]=bad;torch.save(v,candidate)
            try:l.restore_new(candidate,False)
            except AssertionError as err:assert field in str(err)
            else:raise AssertionError('BLOCKED_RESTORE_ACCEPTS_WRONG_LOCK')
            v[field]=original_value
        candidate.unlink()
        rows.append(dict(dataset=l.name,seed=l.seed,strict_parent=True,teacher_independent=True,
                         new_checkpoint_roundtrip=True,wrong_beta_rejected=True,steps=l.tasks[3]['steps']))
        write_json(r.pub/f'ACTUAL_CONFIG_{l.name}_{l.seed}.json',dict(l.args,device=['cuda:0'],FD_beta=1.,only_task=4))
        del l,p,v;gc.collect();torch.cuda.empty_cache();r.resources()
    assert len(rows)==6 and sum(x['steps'] for x in rows)==2750
    write_json(r.pub/'ENGINEERING_GATE.json',dict(status='PASS',parents=rows,coefficient_check=tests,neural_steps=0))


def train(r,parents):
    assert read(r.pub/'ENGINEERING_GATE.json')['status']=='PASS'
    assert not (r.pub/'TRAINING_STARTED.json').exists()
    write_json(r.pub/'TRAINING_STARTED.json',dict(status='RUNNING',unix=time.time(),source_commit=r.cfg['source_commit']))
    r.phase='formal'
    for e in parents:
        l,p=fork(r,e);del p;r.permit(l.name,l.seed,4);l.task_setup(3)
        assert l.teacher_hash==e['network_sha256']
        l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
        finish_task(r,l,l.tasks[3]['steps'],l.steps)
        del l;gc.collect();torch.cuda.empty_cache()
    assert len(r.lineage)==6 and len(r.entries)==12
    assert r.access['formal_optimizer_steps']==2750 and r.access['formal_task_epochs']==60
    write_json(r.pub/'STATE_W_LOCK.json',dict(status='LOCKED',units=r.entries,checkpoints=r.lineage,source_commit=r.cfg['source_commit']))


def evaluate(r):
    lock=read(r.pub/'STATE_W_LOCK.json');assert len(lock['units'])==12 and len(lock['checkpoints'])==6
    assert not (r.pub/'PREDICTIONS_LOCK.json').exists();units=[]
    for e in lock['checkpoints']:
        r.exchange('GET',e['name'],e['sha256']);p=r.private/'parent.part';assert sha(p)==e['sha256']
        l=Student(r,e['dataset'],e['seed']);l.restore_new(p,False)
        assert (l.task,l._total_classes)==(3,8) and network_hash(l._network)==e['network_sha256']
        r.permit(l.name,l.seed,4);raw,_,_,y,rows=r.extract(l,range(8),'val');z=joint(raw);order=np.array(l.order[:8])
        assert network_hash(l._network)==e['network_sha256']
        for method in ('B1A','B1T'):
            q=next(v for v in lock['units'] if (v['dataset'],v['seed'],v['method'])==(l.name,l.seed,method))
            wp=r.private/'banks'/q['W_file'];assert sha(wp)==q['W_sha256']
            with np.load(wp) as f:scores=z@f['W']
            assert np.isfinite(scores).all();name=f'{l.name}_{l.seed}_{method}_t04.npz'
            npz(r.private/'sealed'/name,raw=scores,y=y,original=order[y],order=order,
                ids=np.array([v['sample_id'] for v in rows]),component=np.array([v['identity_component'] for v in rows]))
            units.append(dict(q,file=name,sha256=sha(r.private/'sealed'/name)))
        p.unlink();del l,raw,z;gc.collect();torch.cuda.empty_cache();r.resources()
    assert len(units)==12;write_json(r.pub/'PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units))


def main():
    cfg=read(os.environ['P28_CONFIG']);cfg['mode']=os.environ['P28_MODE'];root=Path(cfg['root']);pub=root/'output/public'
    assert sha(__file__)==read(pub/'PROTOCOL_LOCK.json')['worker_sha256']
    prior=read(pub/'RESOURCE_LEDGER.json') if (pub/'RESOURCE_LEDGER.json').exists() else {}
    r=Run(cfg);r.access=prior.get('calls',{});r.prior_gpu=prior.get('GPU_process_residence_seconds',0)
    r.prior_fit=prior.get('fit_image_reads',0);r.prior_val=prior.get('val_image_reads',0);r.archived=prior.get('archive_bytes',0);r.setup()
    def protect(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
            if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                assert not any(p.is_relative_to(Path(cfg[k]).resolve()) for k in ('ct1_root','ct3p_root','ct4f_root','ct5f_root','ct6f_root'))
    sys.addaudithook(protect)
    original=base.ridge
    def timed(*args,**kwargs):
        t=time.monotonic()
        try:return original(*args,**kwargs)
        finally:r.count('analytic_ms',int(1000*(time.monotonic()-t)))
    base.ridge=timed;torch.save=functools.partial(torch.save,pickle_protocol=4)
    parents=[e for e in read(Path(cfg['ct3p_root'])/'output/public/PARENT_AND_FORK_LINEAGE.json') if e['task']==3]
    try:
        if cfg['mode']=='qualify':qualify(r,parents)
        elif cfg['mode']=='train':train(r,parents)
        elif cfg['mode']=='evaluate':evaluate(r)
        else:raise ValueError(cfg['mode'])
    except BaseException as e:write_json(pub/f'FAILURE_{cfg["mode"]}.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e)));raise
    finally:r.resources(False)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
