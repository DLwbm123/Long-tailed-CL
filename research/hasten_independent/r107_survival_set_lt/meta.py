"""Current-arrival-only pseudo futures, genuine shadow updates and paired credit."""
import random,copy,time
import torch
from memory import Bank,join
from learner import Learner,optimizer,learn,frozen,quality,clone_pair
from manager import Choice,replacement
from rl import update
from common import guard,write

class Meta:
    def __init__(self,c,value,more,actors):
        self.c=c;self.value=value;self.more=more;self.actors=actors
        self.vo=torch.optim.AdamW(value.parameters(),lr=.001,weight_decay=.0001);self.mo=torch.optim.AdamW(more.parameters(),lr=.001,weight_decay=.0001)
        self.actor_opts={k:torch.optim.AdamW(v.parameters(),lr=.001,weight_decay=.0001) for k,v in actors.items() if k!='ZERO_ACTOR'}
    def episode(self,data,rows,stage,number,lt=False):
        seed=self.c['seed']+stage*10000+number*101;rng=random.Random(seed);cap=[4,8,16][number%3];requested=[16,32,64][(number//3)%3];classes=sorted(set(data.y.tolist()));parts=[]
        for cl in classes:
            ix=data.parents[data.y[data.gidx]==cl].cpu().tolist();groups={}
            for i in ix:groups.setdefault(rows[i]['identity_component'],[]).append(i)
            groups=list(groups.values());rng.shuffle(groups)
            if len(groups)<2:continue
            nq=min(4,max(1,len(groups)//4));qi=[i for g in groups[:nq] for i in g];si=[i for g in groups[nq:] for i in g];rng.shuffle(qi);rng.shuffle(si);parts.append((cl,si,qi[:8]))
        if len(parts)<3:return None
        if number%2:rng.shuffle(parts)
        K=len(parts);a=2 if K>=4 else 1;b=a+1
        planned=[]
        for j,(_,ss,_) in enumerate(parts):
            phase=0 if j<a else 1 if j<b else 2
            if not lt:n=cap
            elif number%2==0:n=[16,8,2 if (number//2)%2 else 4][phase]
            else:
                # Paired high/low supports within multi-class phases; singleton role alternates.
                position=j if phase==0 else j-a if phase==1 else j-b
                n=16 if (position+number//2+phase)%2==0 else 4
            planned.append(n)
        parts=[(cl,ss[:n],qq) for (cl,ss,qq),n in zip(parts,planned)]
        si=[i for _,ss,qq in parts for i in ss];qi=[i for _,ss,qq in parts for i in qq];assert not {rows[i]['identity_component'] for i in si}&{rows[i]['identity_component'] for i in qi}
        mapping=torch.full((23,),-1,device=data.z.device,dtype=torch.long)
        for j,(cl,_,_) in enumerate(parts):mapping[cl]=j
        def pick(ix):
            b=data.parents_only(torch.tensor(ix,device=data.z.device));return Bank(b.z,b.pid,mapping[b.y],b.view)
        support=pick(si);query=pick(qi);K=len(parts);a=2 if K>=4 else 1;b=a+1;D=[support.take(support.y<a),support.take((support.y>=a)&(support.y<b)),support.take(support.y>=b)]
        counts=[len(ss) if lt else int((data.y[data.gidx]==cl).sum()) for cl,ss,_ in parts];birth=[0]*a+[1]+[2]*(K-b);B=min(requested,max(a+2,len(D[0])//2))
        return {'lt':lt,'regime':('decreasing' if number%2==0 else 'mixed') if lt else 'legacy','support_counts':[len(ss) for _,ss,_ in parts],'planned_caps':planned,'pseudo_tail':[n<=4 for n in counts] if lt else [n<250 for n in counts],'seed':seed,'D':D,'Q':query,'C':[a,b,K],'counts':counts,'birth':birth,'B':B,'component_intersection':0,'support_parents':len(si),'query_parents':len(qi)}
    def prefix(self,e,counter,kind):
        m=Learner(e['seed']).cuda();m.forbidden=set(e['Q'].pid.tolist());o=optimizer(m);m.arrive(e['D'][0],e['C'][0]);learn(m,o,e['D'][0],12,e['seed']+1,counter,batch=32,kind=kind);return m,o
    def rollout(self,e,prefix,po,actor,role,g,counter,kind):
        m,o=clone_pair(prefix,po,e['seed']);D=e['D'];C=e['C'];ng=e['counts'];born=e['birth'];gen=torch.Generator(device='cuda').manual_seed(e['seed']+100+g);trace=[];infos=[]
        ch=Choice(D[0],m,ng[:C[0]],born[:C[0]],0,e['B'],self.value);M0,tr,info=ch.construct('POLICY',actor,gen,True);trace+=tr;infos.append(info)
        learn(m,o,M0,4,e['seed']+2,counter,batch=32,kind=kind);_,q0=quality(m,e['Q'],counter,kind);teacher=frozen(m);m.arrive(D[1],C[1]);pool=join(M0,D[1]);pool.assert_subset(join(M0,D[1]));learn(m,o,pool,12,e['seed']+3,counter,teacher,32,kind)
        ch=Choice(pool,m,ng[:C[1]],born[:C[1]],1,e['B'],self.value);M1,tr,info=ch.construct('POLICY',actor,gen,True);trace+=tr;infos.append(info);learn(m,o,M1,4,e['seed']+4,counter,teacher,32,kind);u1,q1=quality(m,e['Q'],counter,kind)
        tail=torch.tensor(e['pseudo_tail']);old1=tail[:C[0]];F=[]
        if old1.any():F.append(float((q0-q1[:C[0]]).clamp_min(0)[old1].mean()))
        qb=torch.cat([q0,q1[C[0]:]]);teacher=frozen(m);m.arrive(D[2],C[2]);learn(m,o,join(M1,D[2]),16,e['seed']+5,counter,teacher,32,kind);u2,q2=quality(m,e['Q'],counter,kind);old2=tail[:C[1]]
        if old2.any():F.append(float((qb-q2[:C[1]]).clamp_min(0)[old2].mean()))
        R=(u1+u2)/2-.25*(sum(F)/len(F) if F else 0.)
        return R,trace,{'role':role,'group_member':g,'return':R,'future_utility':[u1,u2],'old_tail_penalty':sum(F)/len(F) if F else None,'memory':infos,'trajectory_length':len(trace)}
    def cf(self,e,prefix,po,index,counter,kind):
        g=torch.Generator(device='cuda').manual_seed(e['seed']+900+index);ch=Choice(e['D'][0],prefix,e['counts'][:e['C'][0]],e['birth'][:e['C'][0]],0,e['B'],self.value);base,_,_=ch.construct('RANDOM',generator=g);alt,i,j=replacement(ch,base,g);values=[]
        for memory in [base,alt]:
            m,o=clone_pair(prefix,po,e['seed']);learn(m,o,memory,4,e['seed']+2,counter,batch=32,kind=kind);teacher=frozen(m);m.arrive(e['D'][1],e['C'][1]);learn(m,o,join(memory,e['D'][1]),12,e['seed']+3,counter,teacher,32,kind);u,_=quality(m,e['Q'],counter,kind);values.append(u)
        d=values[0]-values[1];return torch.stack([ch.raw[i],ch.raw[j]]).detach(),torch.tensor([d,-d],device='cuda'),{'d':d,'utilities':values,'same_initial_model_optimizer_seed_queries_budget':True,'equal_bytes':True,'horizon':1}
    def fit_value(self,m,o,x,y,counter,kind):
        losses=[]
        for _ in range(4):
            o.zero_grad(set_to_none=True);loss=(m(x)-y).square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);o.step();counter[kind+'_value_optimizer']=counter.get(kind+'_value_optimizer',0)+1;losses.append(float(loss.detach()))
        return losses[-1]
    def group(self,e,counter,kind,legacy=None):
        m,o=self.prefix(e,counter,kind);traces={};returns={};diag=[];prefixes={'LT':(m,o)}
        if legacy is not None:prefixes['SET']=self.prefix(legacy,counter,kind)
        for role in ['GSPO_SET_LT','REWARD_SHUFFLED','GSPO_SET']:
            ep=legacy if role=='GSPO_SET' and legacy is not None else e
            pm,po=prefixes['SET'] if role=='GSPO_SET' and legacy is not None else prefixes['LT']
            tt=[];rr=[]
            for g in range(4):
                R,tr,info=self.rollout(ep,pm,po,self.actors[role],role,g,counter,kind);tt.append(tr);rr.append(R);diag.append(info)
                counter[kind+'_trajectory_rollouts']=counter.get(kind+'_trajectory_rollouts',0)+1;counter[kind+'_sampled_actions']=counter.get(kind+'_sampled_actions',0)+len(tr)
            traces[role]=tt;returns[role]=rr
        pairs=[self.cf(e,m,o,k,counter,kind) for k in range(6)];stats={}
        import itertools
        derangements=[p for p in itertools.permutations(range(4)) if all(i!=j for i,j in enumerate(p))]
        perm=derangements[random.Random(e['seed']+61000).randrange(len(derangements))]
        for role in ['GSPO_SET_LT','REWARD_SHUFFLED','GSPO_SET']:
            rr=returns[role] if role!='REWARD_SHUFFLED' else [returns[role][j] for j in perm]
            if kind=='native':
                before=counter.get('policy_optimizer',0);stats[role]=update(self.actors[role],self.actor_opts[role],traces[role],rr,'GSPO',counter);n=counter.pop('policy_optimizer')-before;counter['native_policy_optimizer']=counter.get('native_policy_optimizer',0)+n
            else:stats[role]=update(self.actors[role],self.actor_opts[role],traces[role],rr,'GSPO',counter)
        stats['shared_value_loss']=self.fit_value(self.value,self.vo,*pairs[0][:2],counter,kind)
        stats['more_value_losses']=[self.fit_value(self.more,self.mo,*p[:2],counter,kind) for p in pairs]
        compact={role:[[{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in st.items() if k!='x'} for st in tr] for tr in tt] for role,tt in traces.items()}
        keys=['seed','C','counts','birth','B','component_intersection','support_parents','query_parents','support_counts','planned_caps','pseudo_tail','regime']
        return {'trace':compact,'diagnostics':diag,'counterfactual':[p[2] for p in pairs],'updates':stats,'reward_derangement':perm,'episode':{k:e[k] for k in keys},'legacy_episode':{k:legacy[k] for k in keys} if legacy is not None else None}
    def stage(self,data,rows,t,counter,folder):
        records=[]
        for number in range(16):
            guard(self.c);e=self.episode(data,rows,t,number,True);legacy=self.episode(data,rows,t,number,False)
            if e is None or legacy is None:records.append({'episode':number,'skipped':'fewer_than_three_classes_with_disjoint_components'});continue
            start=time.time();out=self.group(e,counter,'shadow',legacy);torch.save(out.pop('trace'),folder/f'TRACE_e{number}.private.pt');out['group_wall_seconds']=time.time()-start;records.append(out);write(folder/'META_ACTIVE.private.json',{'real_stage':t,'episode':number,'counts':counter})
        write(folder/'META_DIAGNOSTICS.private.json',records);return records
