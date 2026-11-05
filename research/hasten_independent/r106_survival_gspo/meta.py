"""Current-arrival-only pseudo futures, genuine shadow updates and paired credit."""
import random,copy,time
import torch
from memory import Bank,join
from learner import Learner,optimizer,learn,frozen,quality,clone_pair
from manager import Choice,replacement
from rl import update
from common import guard,write

class Meta:
    def __init__(self,c,value,more,gspo,pg):
        self.c=c;self.value=value;self.more=more;self.gspo=gspo;self.pg=pg
        self.vo=torch.optim.AdamW(value.parameters(),lr=.001,weight_decay=.0001);self.mo=torch.optim.AdamW(more.parameters(),lr=.001,weight_decay=.0001)
        self.go=torch.optim.AdamW(gspo.parameters(),lr=.001,weight_decay=.0001);self.po=torch.optim.AdamW(pg.parameters(),lr=.001,weight_decay=.0001)
    def episode(self,data,rows,stage,number):
        seed=self.c['seed']+stage*10000+number*101;rng=random.Random(seed);cap=[4,8,16][number%3];requested=[16,32,64][(number//3)%3];classes=sorted(set(data.y.tolist()));parts=[]
        for cl in classes:
            ix=data.parents[data.y[data.gidx]==cl].cpu().tolist();groups={}
            for i in ix:groups.setdefault(rows[i]['identity_component'],[]).append(i)
            groups=list(groups.values());rng.shuffle(groups)
            if len(groups)<2:continue
            nq=min(4,max(1,len(groups)//4));qi=[i for g in groups[:nq] for i in g];si=[i for g in groups[nq:] for i in g];rng.shuffle(qi);rng.shuffle(si);parts.append((cl,si[:cap],qi[:8]))
        if len(parts)<3:return None
        if number%2:rng.shuffle(parts)
        si=[i for _,ss,qq in parts for i in ss];qi=[i for _,ss,qq in parts for i in qq];assert not {rows[i]['identity_component'] for i in si}&{rows[i]['identity_component'] for i in qi}
        mapping=torch.full((23,),-1,device='cuda',dtype=torch.long)
        for j,(cl,_,_) in enumerate(parts):mapping[cl]=j
        def pick(ix):
            b=data.parents_only(torch.tensor(ix,device='cuda'));return Bank(b.z,b.pid,mapping[b.y],b.view)
        support=pick(si);query=pick(qi);K=len(parts);a=2 if K>=4 else 1;b=a+1;D=[support.take(support.y<a),support.take((support.y>=a)&(support.y<b)),support.take(support.y>=b)]
        counts=[int((data.y[data.gidx]==cl).sum()) for cl,_,_ in parts];birth=[0]*a+[1]+[2]*(K-b);B=min(requested,max(a+2,len(D[0])//2))
        return {'seed':seed,'D':D,'Q':query,'C':[a,b,K],'counts':counts,'birth':birth,'B':B,'component_intersection':0,'support_parents':len(si),'query_parents':len(qi)}
    def prefix(self,e,counter,kind):
        m=Learner(e['seed']).cuda();m.forbidden=set(e['Q'].pid.tolist());o=optimizer(m);m.arrive(e['D'][0],e['C'][0]);learn(m,o,e['D'][0],12,e['seed']+1,counter,batch=32,kind=kind);return m,o
    def rollout(self,e,prefix,po,actor,role,g,counter,kind):
        m,o=clone_pair(prefix,po,e['seed']);D=e['D'];C=e['C'];ng=e['counts'];born=e['birth'];gen=torch.Generator(device='cuda').manual_seed(e['seed']+100+g);trace=[];infos=[]
        ch=Choice(D[0],m,ng[:C[0]],born[:C[0]],0,e['B'],self.value);M0,tr,info=ch.construct('POLICY',actor,gen,True);trace+=tr;infos.append(info)
        learn(m,o,M0,4,e['seed']+2,counter,batch=32,kind=kind);_,q0=quality(m,e['Q'],counter,kind);teacher=frozen(m);m.arrive(D[1],C[1]);pool=join(M0,D[1]);pool.assert_subset(join(M0,D[1]));learn(m,o,pool,12,e['seed']+3,counter,teacher,32,kind)
        ch=Choice(pool,m,ng[:C[1]],born[:C[1]],1,e['B'],self.value);M1,tr,info=ch.construct('POLICY',actor,gen,True);trace+=tr;infos.append(info);learn(m,o,M1,4,e['seed']+4,counter,teacher,32,kind);u1,q1=quality(m,e['Q'],counter,kind)
        tail=torch.tensor(ng)<250;old1=tail[:C[0]];F=[]
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
    def group(self,e,counter,kind):
        m,o=self.prefix(e,counter,kind);traces={};returns={};diag=[]
        for role,actor in [('GSPO',self.gspo),('PG',self.pg)]:
            tt=[];rr=[]
            for g in range(4):
                R,tr,info=self.rollout(e,m,o,actor,role,g,counter,kind);tt.append(tr);rr.append(R);diag.append(info);counter[kind+'_trajectory_rollouts']=counter.get(kind+'_trajectory_rollouts',0)+1;counter[kind+'_sampled_actions']=counter.get(kind+'_sampled_actions',0)+len(tr)
            traces[role]=tt;returns[role]=rr
        pairs=[self.cf(e,m,o,k,counter,kind) for k in range(6)]
        stats={}
        for role,actor,opt in [('GSPO',self.gspo,self.go),('PG',self.pg,self.po)]:
            if kind=='native':
                before=counter.get('policy_optimizer',0);stats[role]=update(actor,opt,traces[role],returns[role],role,counter);n=counter.pop('policy_optimizer')-before;counter['native_policy_optimizer']=counter.get('native_policy_optimizer',0)+n
            else:stats[role]=update(actor,opt,traces[role],returns[role],role,counter)
        stats['shared_value_loss']=self.fit_value(self.value,self.vo,*pairs[0][:2],counter,kind)
        stats['more_value_losses']=[self.fit_value(self.more,self.mo,*p[:2],counter,kind) for p in pairs]
        compact={role:[[{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in st.items() if k!='x'} for st in tr] for tr in tt] for role,tt in traces.items()}
        return {'trace':compact,'diagnostics':diag,'counterfactual':[p[2] for p in pairs],'updates':stats,'episode':{k:e[k] for k in ['seed','C','counts','birth','B','component_intersection','support_parents','query_parents']}}
    def stage(self,data,rows,t,counter,folder):
        records=[]
        for number in range(16):
            guard(self.c);e=self.episode(data,rows,t,number)
            if e is None:records.append({'episode':number,'skipped':'fewer_than_three_classes_with_disjoint_components'});continue
            start=time.time();out=self.group(e,counter,'shadow');__import__('torch').save(out.pop('trace'),folder/f'TRACE_e{number}.private.pt');out['group_wall_seconds']=time.time()-start;records.append(out);write(folder/'META_ACTIVE.private.json',{'real_stage':t,'episode':number,'counts':counter})
        write(folder/'META_DIAGNOSTICS.private.json',records);return records
