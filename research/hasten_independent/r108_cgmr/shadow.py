"""Own-accessible-history pseudo roots; full paired feedback from fresh shadow learners."""
import random,math
import torch
from memory import Bank,join
from roles import relabel
from selection import FixedSelect
from proposals import generate,key
from features import describe,scarce
from learner import Learner,optimizer,learn,frozen,clone_pair
from common import guard

def gen(seed,device):return torch.Generator(device=device).manual_seed(seed)

def episode(pool,roles,rows,observed,seed,number,native_max=False):
 split=roles.split(pool,rows);S=split['S'];classes=sorted(set(S.y.tolist()));eligible=[c for c in classes if bool((split['FIT'].y==c).any()) and bool((split['AUDIT'].y==c).any())]
 if len(eligible)<3:return None,{'skip':'fewer_than_three_jointly_evaluable_classes','support_classes':len(classes),'jointly_evaluable_classes':len(eligible)}
 rng=random.Random(seed);rng.shuffle(eligible);future=eligible[:2];prefix=[c for c in classes if c not in future];regime='decreasing' if number%2==0 else 'mixed'
 ranked=sorted(classes,key=lambda c:(-observed[c],c));caps={c:[32,8,2][min(2,j*3//len(ranked))] for j,c in enumerate(ranked)}
 if native_max:caps={c:32 for c in classes}
 if regime=='mixed':
  values=[caps[c] for c in classes];rng.shuffle(values);caps=dict(zip(classes,values));rng.shuffle(prefix)
 else:prefix.sort(key=lambda c:(-observed[c],c))
 order=prefix+future;mapping=[-1]*23
 for j,c in enumerate(order):mapping[c]=j
 support_ids=[];counts=[]
 for c in order:
  ids=S.parents[S.y[S.gidx]==c].tolist();rng.shuffle(ids);ids=ids[:caps[c]];support_ids+=ids;counts.append(len(ids))
 assert all(n>0 for n in counts)
 support=relabel(S.parents_only(torch.tensor(support_ids,device=pool.z.device)),mapping);queries={};qrng=random.Random(seed+400000)
 for role in ['FIT','AUDIT']:
  bank=split[role];ids=[]
  for c in order:
   a=bank.parents[bank.y[bank.gidx]==c].tolist();qrng.shuffle(a);ids+=a[:8]
  queries[role]=relabel(bank.parents_only(torch.tensor(ids,device=pool.z.device,dtype=torch.long)),mapping)
 si={rows[i]['identity_component'] for i in support.pid.tolist()};fi={rows[i]['identity_component'] for i in queries['FIT'].pid.tolist()};ai={rows[i]['identity_component'] for i in queries['AUDIT'].pid.tolist()};assert not si&fi and not si&ai and not fi&ai
 K0=len(prefix);nphase=min(5,K0);sizes=[K0//nphase+(i<K0%nphase) for i in range(nphase)];ends=[];s=0
 for n in sizes:s+=n;ends.append(s)
 D=[];lo=0;birth=[]
 for j,hi in enumerate(ends):D.append(support.take((support.y>=lo)&(support.y<hi)));birth +=[j]*(hi-lo);lo=hi
 D+=[support.take(support.y==K0),support.take(support.y==K0+1)];birth +=[nphase,nphase+1]
 e={'seed':seed,'requested_B':[64,128,256][number%3],'K0':K0,'K':len(order),'D':D,'Q':queries,'counts':counts,'birth':birth,'prefix_ends':ends,'prefix_phases':nphase,'regime':regime,'component_intersections':[0,0,0]}
 e['coverage']={'support_classes':len(order),'jointly_evaluable_classes':len(eligible),'support_parents':len(support.parents),'fit_query_parents':len(queries['FIT'].parents),'audit_query_parents':len(queries['AUDIT'].parents),'fit_query_local_missing_parents':int(((queries['FIT'].bags()[0][:,-1])==0).sum()),'audit_query_local_missing_parents':int(((queries['AUDIT'].bags()[0][:,-1])==0).sum()),'actual_support_counts':counts,'planned_support_caps':[caps[c] for c in order],'requested_B':e['requested_B'],'C0':K0,'prefix_phases':nphase,'regime':regime}
 return e,None

def root(e,counter,kind,cfg,native=False):
 device=e['D'][0].z.device;m=Learner(e['seed']).to(device);m.forbidden=set(e['Q']['FIT'].pid.tolist())|set(e['Q']['AUDIT'].pid.tolist());o=optimizer(m);memory=None;pool=None
 for j,C in enumerate(e['prefix_ends']):
  guard(cfg);D=e['D'][j];teacher=frozen(m) if m.C else None;pool=join(memory,D);m.arrive(D,C);steps=64//e['prefix_phases']+int(j<64%e['prefix_phases']);learn(m,o,pool,steps,e['seed']+10000+j,counter,teacher,32,kind)
  if j+1<e['prefix_phases']:memory,_,_=FixedSelect(pool,C,e['requested_B']).choose('ZERO',gen(e['seed']+20000+j,device))
 counts=e['counts'][:m.C];birth=e['birth'][:m.C];stage=e['prefix_phases']-1;select=FixedSelect(pool,m.C,e['requested_B']);base,cover,info=select.choose('BALANCED' if native else 'ZERO',gen(e['seed']+30000,device));mems,edits,audit=generate(pool,base,cover,gen(e['seed']+31000,device),counter);x,fa=describe(pool,mems,edits,m,counts,birth,stage,e['D'][e['prefix_phases']-1].parents.tolist());assert not set(pool.pid.tolist())&m.forbidden
 return {'m':m,'o':o,'pool':pool,'mems':mems,'edits':edits,'proposal_audit':audit,'x':x,'feature_audit':fa,'T0':scarce(counts),'actual_B':len(base),'C0':m.C,'info':info,'candidate_keys':[key(M).cpu().tolist() for M in mems],'anchor_keys':key(pool)[cover].cpu().tolist()}

@torch.no_grad()
def query_score(m,q,counter,kind):
 x,y,_=q.bags();mask=y<m.C;x=x[mask];y=y[mask];classes=sorted(set(y.tolist()));assert classes
 p=m(x).double().softmax(-1);good=p.argmax(-1)==y;brier=(p-torch.nn.functional.one_hot(y,m.C)).square().sum(-1);quality=1-brier/2;rec={};qc=[];ba=[]
 for c in classes:
  mm=y==c;correct=int(good[mm].sum());count=int(mm.sum());qv=float(quality[mm].mean());ba.append(correct/count);qc.append(qv);rec[c]={'correct':correct,'count':count,'q':qv}
 counter[kind+'_query_forward_rows']=counter.get(kind+'_query_forward_rows',0)+len(y)
 return {'U':.5*sum(ba)/len(ba)+.5*sum(qc)/len(qc),'classes':rec}

def branch(e,ctx,memory,counter,kind,cfg):
 m,o=clone_pair(ctx['m'],ctx['o'],e['seed']);p=e['prefix_phases'];Q=e['Q'];out={r:[] for r in Q};learn(m,o,memory,4,e['seed']+40000,counter,batch=32,kind=kind);teacher=frozen(m);D=e['D'][p];m.arrive(D,e['K0']+1);pool=join(memory,D);learn(m,o,pool,12,e['seed']+40001,counter,teacher,32,kind)
 for role,q in Q.items():out[role].append(query_score(m,q,counter,kind))
 memory,_,_=FixedSelect(pool,m.C,e['requested_B']).choose('ZERO',gen(e['seed']+50000,pool.z.device));learn(m,o,memory,4,e['seed']+40002,counter,teacher,32,kind);teacher=frozen(m);D=e['D'][p+1];m.arrive(D,e['K']);learn(m,o,join(memory,D),16,e['seed']+40003,counter,teacher,32,kind)
 for role,q in Q.items():out[role].append(query_score(m,q,counter,kind))
 guard(cfg);return out

def feedback(e,ctx,counter,kind,cfg):
 raw=[branch(e,ctx,M,counter,kind,cfg) for M in ctx['mems']];scores={}
 for role in ['FIT','AUDIT']:
  base=raw[0][role];rs=[];us=[];hs=[];valid=[]
  for candidate in raw:
   penalties=[]
   for h in range(2):
    q=candidate[role][h]['classes'];q0=base[h]['classes'];assert set(q)==set(q0);indices=[c for c in q0 if c<e['K0'] and bool(ctx['T0'][c])]
    if indices:penalties.append(sum(max(0.,q0[c]['q']-q[c]['q']) for c in indices)/len(indices))
   U=sum(x['U'] for x in candidate[role])/2;H=sum(penalties)/len(penalties) if penalties else 0.;us.append(U);hs.append(H);rs.append(U-.25*H);valid.append(len(penalties))
  scores[role]={'R':rs,'U':us,'H':hs,'old_scarce_evaluable':bool(valid[0]),'old_scarce_valid_phases':valid[0]}
 counter[kind+'_roots']=counter.get(kind+'_roots',0)+1;counter[kind+'_candidate_branches']=counter.get(kind+'_candidate_branches',0)+len(raw)
 return scores,raw
