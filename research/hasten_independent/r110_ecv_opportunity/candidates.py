"""P1 pool: 12 local/anchor slots plus two structural/random matched pairs."""
import numpy as np
import torch
from .base.anchor_candidates import anchors,_edit
from .base.contracts import Candidate,token_keys,validate_candidate


def _arrays(state,anchor):
    p=state.pool
    original=set(token_keys(anchor));keys=(p.pid*8+p.view).cpu().numpy()
    parent=p.parent_index.cpu().numpy();views=p.view.cpu().numpy();labels=p.y.cpu().numpy()
    gidx=p.gidx.cpu().numpy();gy=labels[gidx]
    local=[np.where((parent==j)&(views>0))[0] for j in range(len(gidx))]
    required=[]
    for c in range(len(state.observed)):
        ix=p.gidx[p.y[p.gidx]==c];center=torch.nn.functional.normalize(p.z[ix].mean(0),dim=0)
        required.append(int(torch.where(p.gidx==ix[(p.z[ix]@center).argmax()])[0]))
    return p,original,keys,parent,views,labels,gidx,gy,local,required


def structural(state,anchor,description):
    p,original,keys,parent,views,labels,gidx,gy,local,required=_arrays(state,anchor)
    z=description['z'].detach().cpu().numpy();chosen_tokens=[]
    features=p.z.detach().cpu().numpy();similarity=(features*features[gidx[parent]]).sum(1)
    for c in range(len(state.observed)):
        qg=int(((anchor.y==c)&(anchor.view==0)).sum());ql=int(((anchor.y==c)&(anchor.view>0)).sum())
        candidates=np.where(gy==c)[0].tolist();selected=[required[c]]
        while len(selected)<qg:
            feasible=[];remaining=qg-len(selected)-1;available=[j for j in candidates if j not in selected]
            caps=sorted([len(local[j]) for j in available],reverse=True);top=sum(caps[:remaining]);nextcap=caps[remaining] if remaining<len(caps) else 0
            distance=(1-z[available]@z[selected].T).min(1)
            selected_cap=sum(len(local[k]) for k in selected)
            for pos,j in enumerate(available):
                cap=len(local[j]);other_top=top-max(0,cap-nextcap) if remaining else 0
                capacity=selected_cap+cap+other_top
                if capacity>=ql:
                    feasible.append((float(distance[pos]),-j,j))
            assert feasible,'STRUCTURAL_COMPLETION_FAILURE'
            selected.append(max(feasible)[2])
        chosen_tokens.extend(gidx[selected].tolist());available=[int(i) for j in selected for i in local[j]]
        selected_locals=[];load={j:0 for j in selected}
        while len(selected_locals)<ql:
            options=[i for i in available if i not in selected_locals]
            def score(i):
                # Fixed coverage/novel-view rule; no query or reward input.
                return (-load[parent[i]],-float(similarity[i]),-i)
            i=max(options,key=score);selected_locals.append(i);load[parent[i]]+=1
        chosen_tokens.extend(selected_locals)
    return p.take(torch.tensor(chosen_tokens,device=p.z.device))


def matched_random(state,anchor,structure,seed,attempts=64):
    p,original,keys,parent,views,labels,gidx,gy,local,required=_arrays(state,anchor)
    target=set(token_keys(structure));rng=np.random.default_rng(seed);tokens=[];used_attempts=0
    for c in range(len(state.observed)):
        qg=int(((anchor.y==c)&(anchor.view==0)).sum());ql=int(((anchor.y==c)&(anchor.view>0)).sum())
        old_global=[j for j in np.where(gy==c)[0] if int(keys[gidx[j]]) in original]
        new_global=[j for j in np.where(gy==c)[0] if int(keys[gidx[j]]) not in original]
        keep_g=sum(int(keys[gidx[j]]) in target for j in old_global)
        old_local={int(i) for j in np.where(gy==c)[0] for i in local[j] if int(keys[i]) in original}
        keep_l=sum(int(keys[i]) in target for i in old_local)
        protected=required[c];others=[j for j in old_global if j!=protected]
        for _ in range(attempts):
            used_attempts+=1
            chosen=[protected]+rng.choice(others,keep_g-1,replace=False).tolist()+rng.choice(new_global,qg-keep_g,replace=False).tolist()
            available=[int(i) for j in chosen for i in local[j]]
            left=[i for i in available if i in old_local];right=[i for i in available if i not in old_local]
            if len(left)<keep_l or len(right)<ql-keep_l:continue
            loc=rng.choice(left,keep_l,replace=False).tolist()+rng.choice(right,ql-keep_l,replace=False).tolist()
            tokens.extend(gidx[chosen].tolist()+loc);break
        else:return None,used_attempts
    memory=p.take(torch.tensor(tokens,device=p.z.device))
    assert overlap_profile(memory,anchor)==overlap_profile(structure,anchor)
    return memory,used_attempts


def overlap_profile(memory,anchor):
    a=set(token_keys(anchor));out=[]
    for c in sorted(set(anchor.y.tolist())):
        for local in [False,True]:
            mask=(memory.y==c)&((memory.view>0)==local)
            keys=set((memory.pid[mask]*8+memory.view[mask]).tolist())
            out.append((len(keys),len(keys&a)))
    return out


def generate(state,model,description,seed):
    zero,bal=anchors(state,model,seed);out=[];seen={};audit=[]
    def add(memory,family,name,anchor,matched=None):
        removals=len(set(token_keys(anchor))-set(token_keys(memory)))
        c=Candidate(memory,family,name,removals);validate_candidate(c,state,anchor)
        if c.key in seen:return seen[c.key]
        seen[c.key]=len(out);out.append(c)
        return len(out)-1
    add(zero,'anchor','ZERO',zero);add(bal,'anchor','BAL',bal)
    for ai,(name,anchor) in enumerate([('ZERO',zero),('BAL',bal)]):
        selected=torch.isin(state.pool.parents,anchor.parents);y=state.pool.y[state.pool.gidx];z=description['z'];cover=torch.zeros(len(y),device=z.device)
        for c in range(model.C):
            mask=y==c;cover[mask]=1-(z[mask]@z[mask&selected].T).max(1).values
        for j,(family,limit) in enumerate([('risk',1),('risk',2),('risk',4),('cover',4),('random',2)]):
            priority=(description['parent_risk'] if family=='risk' else cover if family=='cover' else torch.zeros_like(cover))[state.pool.parent_index]
            eligible=description['scored'][state.pool.parent_index] if family=='risk' else None
            memory,count=_edit(state,anchor,priority,min(limit,max(1,len(anchor)//10)),seed+ai*100+j,family=='random',eligible)
            add(memory,family,name,anchor)
    for ai,(name,anchor) in enumerate([('ZERO',zero),('BAL',bal)]):
        structure=structural(state,anchor,description);s=add(structure,'structural',name,anchor)
        random,attempts=matched_random(state,anchor,structure,seed+1000+ai)
        j=None if random is None else add(random,'matched_random',name,anchor)
        audit.append(dict(anchor=name,structural_index=s,random_index=j,attempts=attempts,matched=random is not None,distinct=j!=s if j is not None else False))
    assert 1<=len(out)<=16 and out[0].anchor=='ZERO'
    return out,audit

