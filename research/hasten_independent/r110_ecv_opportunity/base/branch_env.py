"""Isolated pulse branches of the real R107 inner operator.

Pseudo-arrivals are subsets of the already legal root evidence. They are not
permission to access later real stages. No branch can recover discarded support.
"""
import copy
import random
import numpy as np
import torch
from .learner import clone_pair, learn, frozen
from .memory import join
from .contracts import VisibleState, token_keys
from .anchor_candidates import anchors
from .paired_labels import evaluate, paired, utility


class Snapshot:
    def __init__(self, model, optimizer, scheduler=None):
        self.model, self.optimizer, self.scheduler = copy.deepcopy((model,optimizer,scheduler))
        self.torch_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.numpy_rng = np.random.get_state()
        self.python_rng = random.getstate()

    def restore(self):
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        np.random.set_state(self.numpy_rng); random.setstate(self.python_rng)
        return copy.deepcopy((self.model,self.optimizer,self.scheduler))


def event_seed(root_seed, event):
    import hashlib
    return int.from_bytes(hashlib.blake2b(f'{root_seed}:{event}'.encode(),digest_size=8).digest(),'little') % (2**63-1)


def paired_learn(model, opt, bank, steps, seed, counter, teacher=None, batch=64):
    """Same AdamW/CE/KD law, event+parent-key common randomness.

    Per-slot hash priorities couple overlapping parents by identity rather than
    matching unrelated samples at the same bank position. This is not bitwise
    replay of the legacy generator; formal deployment keeps legacy learn().
    """
    assert not set(bank.pid.tolist()) & getattr(model,'forbidden',set())
    x,y,pids = bank.bags()
    parents = pids.cpu().numpy().astype(np.uint64)
    labels = y.cpu().numpy()
    assert set(labels.tolist()) == set(range(model.C))
    rng = np.random.default_rng(seed)
    for step in range(steps):
        cls = rng.integers(model.C,size=batch)
        salts = rng.integers(0,np.iinfo(np.uint64).max,size=batch,dtype=np.uint64)
        index = np.empty(batch,dtype=np.int64)
        for c in range(model.C):
            slots = np.where(cls==c)[0]
            if not len(slots):
                continue
            positions = np.where(labels==c)[0]
            h = parents[positions][None,:] ^ salts[slots,None]
            with np.errstate(over='ignore'):
                h = (h ^ (h >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
                h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
                h = h ^ (h >> np.uint64(31))
            index[slots] = positions[h.argmin(1)]
        ix = torch.tensor(index,device=x.device)
        xx, yy = x[ix],y[ix]
        opt.zero_grad(set_to_none=True)
        logits = model(xx); loss = torch.nn.functional.cross_entropy(logits,yy)
        if teacher is not None and teacher.C:
            mask = yy < teacher.C
            if mask.any():
                with torch.no_grad():
                    target = (teacher(xx[mask])/2).softmax(-1)
                loss = loss+2*torch.nn.functional.kl_div((logits[mask,:teacher.C]/2).log_softmax(-1),target,reduction='batchmean')
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        counter['paired_classifier_updates'] = counter.get('paired_classifier_updates',0)+1
        counter['paired_training_rows'] = counter.get('paired_training_rows',0)+batch


@torch.no_grad()
def probe(model, bank):
    x,y,_ = bank.bags(); logits=model(x)
    p=logits.softmax(-1); other=logits.clone(); other.scatter_(1,y[:,None],-1e6)
    margin=logits.gather(1,y[:,None]).squeeze(1)-other.max(1).values
    if model.C==1: margin.zero_()
    ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
    brier=(p-torch.nn.functional.one_hot(y,model.C)).square().sum(1)
    out=[]
    for c in range(model.C):
        mask=y==c
        out.append(torch.stack([margin[mask].mean()/40,ce[mask].mean(),brier[mask].mean()]))
    return torch.stack(out).detach()


class PulseEnvironment:
    def __init__(self, model, opt, state, future, queries, oldtail, ledger,
                 counter=None, teacher=None, post_steps=100, pre_steps=200, batch=64):
        state.validate(model)
        assert 1 <= len(queries) <= 3 and len(future) == len(queries)-1
        self.snapshot=Snapshot(model,opt); self.state=state
        self.future=future; self.queries=queries; self.oldtail=set(oldtail)
        self.ledger=ledger; self.counter={} if counter is None else counter
        self.teacher=teacher; self.post_steps=post_steps; self.pre_steps=pre_steps; self.batch=batch
        self.ledger.check(state.pool,'support')
        available=set(token_keys(state.pool))
        for arrival in future:
            self.ledger.check(arrival,'support')
            assert not available & set(token_keys(arrival)), 'REPEATED_PSEUDO_SUPPORT_ARRIVAL'
            available.update(token_keys(arrival))
        for query in queries:
            self.ledger.check(query,'query')
        # Caller must prove the checkpoint never fitted any reward group.
        query_groups={ledger.groups[int(i)] for q in queries for i in q.parents}
        trained=getattr(model,'trained_groups',None)
        assert trained is not None, 'UNKNOWN_CHECKPOINT_TRAINING_PROVENANCE'
        assert not query_groups & set(trained), 'CHECKPOINT_SAW_QUERY_GROUP'

    def rollout(self, first_memory, seed, selector=None):
        model,opt,_=self.snapshot.restore()
        memory=first_memory; memory.assert_subset(self.state.pool)
        forbidden={int(i) for q in self.queries for i in q.parents}
        model.forbidden=forbidden
        endpoints=[]; choices=[]; dynamic=None; C=model.C
        observed=list(self.state.observed); birth=list(self.state.birth)
        old_teacher=self.teacher
        for h,query in enumerate(self.queries):
            if h:
                current=self.future[h-1]; pool=join(memory,current)
                self.ledger.check(pool,'support',token_keys(memory)+token_keys(current))
                old_teacher=frozen(model); C=max(C,int(current.y.max())+1)
                model.arrive(current,C)
                for c in range(len(observed),C):
                    observed.append(int((current.y[current.gidx]==c).sum())); birth.append(self.state.stage+h)
                paired_learn(model,opt,pool,self.pre_steps,event_seed(seed,f'{h}:pre'),self.counter,old_teacher,self.batch)
                visible=VisibleState(pool,memory,tuple(observed),tuple(birth),self.state.stage+h,self.state.capacity,self.state.groups,self.state.independent_groups_verified,self.ledger,old_teacher)
                zero,_=anchors(visible,model,event_seed(seed,f'{h}:zero'))
                if selector is None:
                    memory=zero
                else:
                    memory,record=selector(visible,model,opt,event_seed(seed,f'{h}:policy'))
                    memory.assert_subset(pool); choices.append(record)
            if h==0:
                before=probe(model,memory)
            paired_learn(model,opt,memory,self.post_steps,event_seed(seed,f'{h}:post'),self.counter,old_teacher,self.batch)
            if h==0:
                dynamic=probe(model,memory)-before
            endpoints.append(evaluate(model,query,self.counter))
        return dict(endpoints=endpoints,dynamic=dynamic,model=model,optimizer=opt,memory=memory,choices=choices)

    def compare(self, candidates, seed):
        zero=self.rollout(candidates[0].memory,seed)
        output=[]
        for i,candidate in enumerate(candidates):
            run=zero if i==0 else self.rollout(candidate.memory,seed)
            differences=[paired(a,b,self.oldtail) for a,b in zip(run['endpoints'],zero['endpoints'])]
            changed=any(not torch.equal(v,zero['model'].state_dict()[k]) for k,v in run['model'].state_dict().items())
            output.append(dict(summary=utility(differences),endpoints=differences,dynamic=run['dynamic'],candidate=candidate,parameters_differ_from_zero=changed))
        assert abs(output[0]['summary']['R'])<1e-12
        return output

    def formal_step(self, model, opt, current, previous, selector, observed, birth, stage, seed, ledger):
        """Unmodified 200 -> selection -> 100 deployment ordering and RNG."""
        teacher=frozen(model) if model.C else None
        pool=join(previous,current)
        ledger.check(pool,'support')
        model.arrive(current,len(observed))
        learn(model,opt,pool,200,seed+stage*100+1,self.counter,teacher,64,'formal')
        state=VisibleState(pool,previous,tuple(observed),tuple(birth),stage,256,ledger.groups,False,ledger,teacher)
        memory=selector(state,model,opt,seed+70000+stage)
        memory.assert_subset(pool)
        learn(model,opt,memory,100,seed+stage*100+2,self.counter,teacher,64,'formal')
        return memory

