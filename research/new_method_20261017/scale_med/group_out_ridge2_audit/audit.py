import json,time,gc
from pathlib import Path
import numpy as np
import torch
from solver import mapper,transform
from core_attribution import weights
from train_base import setup
from binding import native
from space import check_space
ARMS=['F','E','I','O'];GRID=[.0001,.01,1.]
def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));t.replace(p)
def points(pred,y):
 rec=[float((pred[y==c]==c).double().mean()) for c in range(4)];return dict(BA=sum(rec)/4,recall=rec,correct=[int(((pred==c)&(y==c)).sum()) for c in range(4)],counts=[int((y==c).sum()) for c in range(4)])
def pairs(base,pred,y):
 rows=[]
 for c in range(4):
  mask=y==c;b=base[mask];p=pred[mask];bo=b==c;po=p==c;r=int((~bo&po).sum());h=int((bo&~po).sum());joint=torch.bincount(b*4+p,minlength=16).reshape(4,4);rows.append(dict(class_index=c,count=len(p),rescued=r,harmed=h,both_correct=int((bo&po).sum()),both_wrong=int((~bo&~po).sum()),net_correct=r-h,baseline_prediction_by_candidate_prediction=joint.tolist(),harmed_destination=torch.bincount(p[bo&~po],minlength=4).tolist(),rescued_from=torch.bincount(b[~bo&po],minlength=4).tolist()))
 delta=100*sum(x['net_correct']/x['count'] for x in rows)/4;assert abs(delta-100*(points(pred,y)['BA']-points(base,y)['BA']))<1e-10;return dict(BA_gain_pp=delta,classes=rows)
def oracle(b,p,y):
 ok=(b==y)|(p==y);r=[float(ok[y==c].double().mean()) for c in range(4)];assert all(int(ok[y==c].sum())>=max(int((b[y==c]==c).sum()),int((p[y==c]==c).sum())) for c in range(4));return dict(BA=sum(r)/4,recall=r,correct=[int(ok[y==c].sum()) for c in range(4)],privileged_fixed_pair_not_deployable=True)
def lists(y,gg):
 gen=torch.Generator().manual_seed(74240);members={c:{g:[i for i in torch.where(y==c)[0].tolist() if gg[i]==g] for g in sorted({gg[i] for i in torch.where(y==c)[0].tolist()})} for c in range(4)};assert min(len(a) for a in members.values())>=8;out={4:[],8:[]}
 for repeat in range(8):
  small=[];large=[]
  for c in range(4):
   keys=list(members[c]);choose=torch.randperm(len(keys),generator=gen)[:8];ids=[]
   for j in choose:
    vals=members[c][keys[int(j)]];ids.append(vals[int(torch.randint(len(vals),(1,),generator=gen))])
   small+=ids[:4];large+=ids
  assert len({gg[i] for i in large})==32 and set(small).issubset(large);out[4].append(torch.tensor(small));out[8].append(torch.tensor(large))
 return out,{str(c):len(members[c]) for c in members}
def dual(x,y,q,alpha):
 w=q/q.sum();mx=w@x;hot=torch.nn.functional.one_hot(y,4).to(x);my=w@hot;xx=(x-mx)*w.sqrt()[:,None];yy=(hot-my)*w.sqrt()[:,None];kernel=xx@xx.T;h=kernel+alpha*torch.eye(len(x),dtype=x.dtype);v=torch.linalg.solve(h,yy);res=float((h@v-yy).abs().max());assert res<=1e-7;W=xx.T@v;b=my-mx@W;e=torch.linalg.eigvalsh(kernel);assert float(e.min())>=-1e-8;df=1+float((e.clamp_min(0)/(e.clamp_min(0)+alpha)).sum());assert 1-1e-8<=df<=min(len(x),x.shape[1]+1)+1e-7;return W,b,df,res

def flags(cells,full):
 get=lambda m,n,a:[v for v in cells if v['arm']=='O' and v['groups_per_class']==m and v['norm']==n and v['alpha']==a];support=[];positive=[];norm=[];persistent=[]
 for m in [4,8]:
  for alpha in GRID:
   vs=get(m,'full',alpha);ref=next(v for v in full if v['arm']=='O' and v['alpha']==alpha)['vs_F'];fullba=ref['BA_gain_pp'];fullc=100*ref['classes'][2]['net_correct']/ref['classes'][2]['count'];ba=np.array([v['vs_F']['BA_gain_pp'] for v in vs]);c2=np.array([100*v['vs_F']['classes'][2]['net_correct']/v['vs_F']['classes'][2]['count'] for v in vs]);support.append(bool(ba.mean()-fullba>=2 and c2.mean()-fullc>=5 and int((ba-fullba>0).sum())>=6));positive.append(bool(ba.mean()>=1 and c2.mean()>=0));small=get(m,'support',alpha);d=ba-np.array([v['vs_F']['BA_gain_pp'] for v in small]);norm.append((float(d.mean()),int((d>0).sum()),int((d<0).sum())))
 for m in [4,8]:
  for n in ['support','full']:
   for alpha in GRID:persistent.append(np.mean([v['vs_F']['classes'][2]['net_correct'] for v in get(m,n,alpha)])<0)
 persistent.extend(v['vs_F']['classes'][2]['net_correct']<0 for v in full if v['arm']=='O');return dict(CONSISTENT_SMALL_SUPPORT_INTERACTION=all(support),SMALL_SUPPORT_POSITIVE_PATTERN=all(support) and all(positive),normalization_interaction='CONSISTENT_FULL_NORM_INTERACTION' if all(d>=2 and p>=6 for d,p,n in norm) else ('CONSISTENT_SUPPORT_NORM_INTERACTION' if all(d<=-2 and n>=6 for d,p,n in norm) else 'MIXED_OR_WEAK'),PERSISTENT_CLASS2_HARM_ACROSS_DESIGN=bool(all(persistent)),not_causal_identification=True)

def synthetic():
 y=torch.tensor([0,0,1,1,2,2,3,3]);b=torch.tensor([0,1,1,0,2,0,3,0]);p=torch.tensor([1,0,0,1,0,2,0,3]);v=pairs(b,p,y);assert v['BA_gain_pp']==0 and v['classes'][2]['rescued']==1 and v['classes'][2]['harmed']==1;assert oracle(b,p,y)['BA']==1
 gen=torch.Generator().manual_seed(74240);x=torch.randn(16,7,generator=gen,dtype=torch.float64);yy=torch.arange(16)%4;q=torch.rand(16,generator=gen,dtype=x.dtype)+.1;W,bb,df,e=dual(x,yy,q,.01);z=torch.cat([x,torch.ones(16,1,dtype=x.dtype)],1);w=q/q.sum();a=z.T@(z*w[:,None]);a.diagonal()[:-1].add_(.01);rhs=z.T@(torch.nn.functional.one_hot(yy,4).double()*w[:,None]);dense=torch.linalg.solve(a,rhs);assert float((z@dense-(x@W+bb)).abs().max())<1e-9;hat=z@torch.linalg.solve(a,z.T*w);assert abs(float(hat.trace())-df)<1e-9
 sy=torch.arange(64)//16;sg=['toy'+str(i) for i in range(64)];ss,gc=lists(sy,sg);assert all(len(t)==4*k for k in ss for t in ss[k]);assert 4*(2*8*2*3+3)==396
 def fake(ba,net):return dict(BA_gain_pp=ba,classes=[{}, {},dict(net_correct=net,count=100),{}])
 full=[dict(arm='O',alpha=a,vs_F=fake(-5,-20)) for a in GRID];good=[];bad=[]
 for k in [4,8]:
  for norm in ['support','full']:
   for a in GRID:
    for rep in range(8):
     common=dict(arm='O',groups_per_class=k,norm=norm,alpha=a,repeat=rep);good.append(dict(**common,vs_F=fake(3 if norm=='full' else 0,0)));bad.append(dict(**common,vs_F=fake(-5,-20)))
 fg=flags(good,full);fb=flags(bad,full);assert fg['CONSISTENT_SMALL_SUPPORT_INTERACTION'] and fg['SMALL_SUPPORT_POSITIVE_PATTERN'] and fg['normalization_interaction']=='CONSISTENT_FULL_NORM_INTERACTION';assert not fb['CONSISTENT_SMALL_SUPPORT_INTERACTION'] and fb['PERSISTENT_CLASS2_HARM_ACROSS_DESIGN'];json.dumps(dict(good=fg,bad=fb),allow_nan=False);return dict(paired_identity_oracle=True,primal_dual=True,df_hat_trace=True,group_nested_schedules=True,expected_logical_units=396,zero_updates=True,seed=74240)

def extract(cfg,arm,obj,out):
 cache=Path(cfg['root'])/(arm+'.g.private.pt');seal=Path(cfg['root'])/(arm+'.FEATURE_SEAL.json');started=Path(cfg['root'])/(arm+'.ENCODING_STARTED.private.json')
 if seal.exists():
  g=torch.load(cache,weights_only=False);assert list(g.shape)==[3236,1536];return g
 assert not started.exists(),'unsealed encoding started: cannot silently exceed one-pass quota';write(started,dict(arm=arm,rows_quota=3236,started_unix=time.time()));m,params,keys,ckeys,heads,pw=setup(cfg);old=torch.load(Path(cfg['previous'])/'train'/f'{arm}.encoder.private.pt',weights_only=False);over=native(old['bias_overrides']);assert sorted(over)==keys
 with torch.no_grad():
  for k in keys:params[k].copy_(over[k].cuda())
  index=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());assert index['rows']==3236;vals=[];rows=0
  for cell in index['chunks']:
   x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False)
   for xx in x.split(32):
    check_space(cfg['window_root']);vals.append(torch.nn.functional.normalize(m(xx.cuda()).cpu().double(),dim=1));rows+=len(xx);write(out/'status.json',dict(state='RUNNING',phase='ONE_PASS_ENDPOINT_G',arm=arm,rows=rows,total_rows=3236))
 g=torch.cat(vals);assert torch.isfinite(g).all() and list(g.shape)==[3236,1536];check_space(cfg['window_root'],reserve=g.numel()*g.element_size()*3);torch.save(g.contiguous(),cache);write(seal,dict(arm=arm,rows=3236,dim=1536,one_pass=True,source_commit=cfg['source_commit'],test_used=False));del m,params,heads,over,old;gc.collect();torch.cuda.empty_cache();return g

def run(cfg):
 out=Path(cfg['output']);assert json.loads(Path(cfg['preflight']).read_text())['pass_all'];obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];groups=obj['groups'];yy=y[va];trainy=y[tr];traing=[groups[i] for i in torch.where(tr)[0].tolist()];assert int(tr.sum())==2555 and int(va.sum())==681 and not set(traing)&{groups[i] for i in torch.where(va)[0].tolist()};q=weights(trainy,traing);schedule,ng=lists(trainy,traing);write(Path(cfg['root'])/'SUPPORT_LISTS.private.json',{str(k):[v.tolist() for v in a] for k,a in schedule.items()});old=json.loads((Path(cfg['previous'])/'fit/aggregate.json').read_text());assert not old['test_used'];expected={a:v['0'] for a,v in old['development'][0]['summary'].items()};g={'F':obj['g'].double()};maps={'F':mapper(g['F'][tr],q)};mat={};heads={};pred={};full=[];a1={};rawquery={};fullx={}
 def prepare(arm):
  if arm!='F':
   g[arm]=extract(cfg,arm,obj,out);st=torch.load(Path(cfg['previous'])/'fit'/f'{arm}.0.state.private.pt',weights_only=False);maps[arm]=st['mapper'];ix=torch.triu_indices(1536,1536);a=torch.zeros(1536,1536,dtype=torch.float64);a[ix.unbind()]=st['A'];a[ix[1],ix[0]]=st['A'];mat[arm]=(a,st['B']);heads[arm]=st['W']
  z=transform(g[arm],maps[arm]);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);fullx[arm]=x[tr].clone();rawquery[arm]=g[arm][va].clone()
  if arm=='F':
   a=x[tr].T@(x[tr]*q[:,None]);a=(a+a.T)/2;b=x[tr].T@(torch.nn.functional.one_hot(trainy,6).double()*q[:,None]);mat[arm]=(a,b);h=a/4;h=h.clone();h.diagonal()[:-1].add_(.01);heads[arm]=torch.linalg.solve(h,b/4)
  v=x[va]@heads[arm][:,:4];pred[arm]=v.argmax(1);assert points(pred[arm],yy)['correct']==expected[arm]['correct'] and abs(points(pred[arm],yy)['BA']-expected[arm]['BA'])<1e-12
  if arm!='F':a1[arm+'_vs_F']=pairs(pred['F'],pred[arm],yy);write(out/'A1_PARTIAL.json',dict(comparisons=a1,completed=list(pred),test_used=False))
  a,b=mat[arm];mu=(q[:,None]*x[tr,:-1]).sum(0)/4;center=a[:-1,:-1]/4-mu[:,None]*mu[None,:];ev=torch.linalg.eigvalsh(center);assert float(ev.min())>=-1e-8;ev=ev.clamp_min(0)
  for alpha in GRID:
   h=a/4;h=h.clone();h.diagonal()[:-1].add_(alpha);w=torch.linalg.solve(h,b/4);res=float((h@w-b/4).abs().max());assert res<=1e-7;pr=(x[va]@w[:,:4]).argmax(1);pt=points(pr,yy);prior=next(v for v in old['fits'] if v['arm']==arm and v['alpha']==alpha);assert abs(pt['BA']-prior['initial_validation_BA'])<1e-12;full.append(dict(arm=arm,alpha=alpha,groups_per_class='full',norm='full',points=pt,df=1+float((ev/(ev+alpha)).sum()),residual=res,private_pred=pr));
  del z,x
 prepare('F')
 for arm in ['O','E','I']:prepare(arm)
 a1['O_vs_I']=pairs(pred['I'],pred['O'],yy);a1['O_vs_E']=pairs(pred['E'],pred['O'],yy);rr=a1['O_vs_F']['classes'][2];write(out/'A1.json',dict(comparisons=a1,oracle_F_O=oracle(pred['F'],pred['O'],yy),summary={a:points(p,yy) for a,p in pred.items()},class2_interpretation='NO_CLASS2_RESCUE_ON_THIS_DEV' if rr['rescued']==0 and rr['harmed']>0 else ('RESCUE_WITH_NET_HARM' if rr['harmed']>rr['rescued']>0 else 'OTHER_OR_MIXED'),method_success=False,test_used=False));torch.save(dict(pred=pred,y=yy,groups=[groups[i] for i in torch.where(va)[0].tolist()]),out/'development_predictions.private.pt')
 for v in full:
  ref=next(x['private_pred'] for x in full if x['arm']=='F' and x['alpha']==v['alpha']);v['vs_F']=pairs(ref,v['private_pred'],yy)
 cells=[]
 for count in [4,8]:
  for rep,ids in enumerate(schedule[count]):
   sq=weights(trainy[ids],[traing[i] for i in ids]);assert abs(float(sq.sum())-4)<1e-10
   for norm in ['support','full']:
    local={}
    for arm in ARMS:
     gt=g[arm][tr][ids];mp=mapper(gt,sq) if norm=='support' else maps[arm];xs=transform(gt,mp);xq=transform(rawquery[arm],mp)
     for alpha in GRID:
      W,b,df,res=dual(xs,trainy[ids],sq,alpha);pr=(xq@W+b).argmax(1);local[arm,alpha]=(pr,df,res)
    for arm in ARMS:
     for alpha in GRID:
      pr,df,res=local[arm,alpha];ref=local['F',alpha][0];fp=next(v['private_pred'] for v in full if v['arm']==arm and v['alpha']==alpha);cells.append(dict(arm=arm,groups_per_class=count,repeat=rep,norm=norm,alpha=alpha,points=points(pr,yy),df=df,residual=res,vs_F=pairs(ref,pr,yy),vs_own_full=pairs(fp,pr,yy)))
    write(out/'status.json',dict(state='RUNNING',phase='A2_CPU_SOLVES',completed=len(cells),required=384));write(out/'A2_PARTIAL.json',dict(cells=cells));check_space(cfg['window_root'])
 assert len(cells)==384 and len(full)==12;fs=[{k:v for k,v in cell.items() if k!='private_pred'} for cell in full];ff=flags(cells,fs);write(out/'A2.json',dict(cells=cells,full_replays=fs,flags=ff,known_training_groups=ng,logical_units=396,selection_from_grid=False,independent_cohorts=1,method_success=False))
 ncm={};npr={}
 for arm in ARMS:
  x=fullx[arm][:,:-1];centers=torch.stack([(x[trainy==c]*q[trainy==c,None]).sum(0)/q[trainy==c].sum() for c in range(4)]);query=transform(rawquery[arm],maps[arm]);dist=query.square().sum(1,keepdim=True)+centers.square().sum(1)[None,:]-2*query@centers.T;npr[arm]=dist.argmin(1);ncm[arm]=points(npr[arm],yy)
 cross={a:pairs(npr['F'],npr[a],yy) for a in ['E','I','O']};ridgegain=a1['O_vs_F']['classes'][2]['net_correct']/38*100;ncmgain=cross['O']['classes'][2]['net_correct']/38*100;flag='CROSS_READOUT_CLASS2_HARM' if ridgegain<=-5 and ncmgain<=-5 else ('RIDGE_INTERACTION_PLAUSIBLE' if ridgegain<=-5 and ncmgain>=0 else 'MIXED_OR_UNRESOLVED');write(out/'A3.json',dict(summary=ncm,vs_F=cross,interpretation=flag,not_information_loss_proof=True,method_success=False));write(out/'aggregate.json',dict(decision='A1_A2_A3_DIAGNOSTICS_COMPLETE',arms=ARMS,feature_rows={a:3236 for a in ['E','I','O']},new_parameter_updates=0,full_replays=12,new_small_support_solves=384,logical_fit_units=396,NCM_units=4,A1_class2=rr,A1_oracle=oracle(pred['F'],pred['O'],yy),A2_flags=ff,A3_class2_gain_pp=ncmgain,A3_flag=flag,test_used=False,future_used=False,method_success=False));write(out/'status.json',dict(state='COMPLETE',decision='A1_A2_A3_DIAGNOSTICS_COMPLETE'))
