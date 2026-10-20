import json,os,time
from pathlib import Path
import torch,numpy as np
from train_base import setup
from orbit import views,mapper,transform,stats,fit,synthetic
from core_attribution import weights
from evaluation import metrics,evaluate
from space import check_space
ARMS=['F','T','M','I','A'];CAP=19058800

def write(p,a):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(a,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));t.replace(p)
def features(m,x):return torch.nn.functional.normalize(m(x.cuda()).cpu().double(),dim=1)
def extract(cfg,m,stage,obj,out):
 index=json.loads((Path(cfg['matched'])/f'pixels_{stage}/INDEX.json').read_text());assert index['rows']==len(obj['y']);vv=[];err=0;rows=0
 with torch.no_grad():
  for cell in index['chunks']:
   x=torch.load(Path(cfg['matched'])/f'pixels_{stage}'/cell['file'],weights_only=False)
   for j,z in enumerate(x.split(32)):
    check_space(cfg['window_root']);v=torch.stack([features(m,xx) for xx in views(z)],1);first=cell['first']+j*32;err=max(err,float((v[:,0]-obj['g'][first:first+len(z)]).abs().max()));vv.append(v);rows+=len(z);write(out/'status.json',dict(state='RUNNING',stage=stage,phase='frozen_D4_forward',rows=rows,total_rows=len(obj['y']),logical_forward_images=rows*8))
 assert err<=1e-10;return torch.cat(vv),err

def table(sc,y):
 return {a:{str(s):{k:val for k,val in metrics(v[y<s+4],y[y<s+4],['unused'+str(i) for i in range(int((y<s+4).sum()))],s+4).items() if k not in ['group_equal_BA','group_recall']} for s,v in stages.items()} for a,stages in sc.items()}
def devgate(tab,stage):
 a,f=tab['A'],tab['F'];delta=lambda s:np.array(a[str(s)]['recall'])-f[str(s)]['recall'];r={}
 if stage==0:
  d=delta(0);r=dict(BA_gain_pp=100*(a['0']['BA']-f['0']['BA']),class1_gain_pp=100*d[1],other_class_gains_pp=(100*d[[0,2,3]]).tolist(),control_BA_gains_pp={c:100*(a['0']['BA']-tab[c]['0']['BA']) for c in ['M','I','T']});r['pass_all']=bool(r['BA_gain_pp']>=1-1e-12 and r['class1_gain_pp']>=5-1e-12 and min(r['other_class_gains_pp'])>=-1-1e-12 and min(r['control_BA_gains_pp'].values())>0)
 elif stage==2:
  J=lambda t:50*(t['1']['BA']+t['2']['BA']);r=dict(J_gain_pp=J(a)-J(f),control_J_gains_pp={c:J(a)-J(tab[c]) for c in ['M','I','T']},class1_gains_pp=[100*delta(s)[1] for s in range(3)],future_BA_gains_pp=[100*(a[str(s)]['BA']-f[str(s)]['BA']) for s in [1,2]],birth_retired_gains_pp=[100*delta(s)[c] for s,c in [(1,4),(2,4),(2,5)]],old_macro_gains_pp=[100*delta(s)[:4].mean() for s in [1,2]]+[100*delta(2)[:5].mean()],class0_gains_pp=[100*delta(s)[0] for s in range(3)]);r['pass_all']=bool(r['J_gain_pp']>=1-1e-12 and min(r['control_J_gains_pp'].values())>=.5-1e-12 and min(r['class1_gains_pp'])>=5-1e-12 and min(r['future_BA_gains_pp']+r['birth_retired_gains_pp'])>=-1e-12 and min(r['old_macro_gains_pp']+r['class0_gains_pp'])>=-1-1e-12)
 else:return None
 return r

def artifact(cfg,out,arm,st,mat,w,mp,alpha):
 aa,bb=mat;ix=torch.triu_indices(len(aa),len(aa));p=out/f'{arm}.{st}.state.private.pt';check_space(cfg['window_root'],reserve=24*2**20);torch.save(dict(A=aa[ix.unbind()].contiguous(),B=bb.contiguous(),W=w.contiguous(),mapper=mp,alpha=alpha,classes=st+4,views='identity' if arm=='F' else 'fixed_D4',source_commit=cfg['source_commit']),p);assert p.stat().st_size<=CAP;c=torch.load(p,weights_only=False);rec=torch.zeros_like(aa);rec[ix.unbind()]=c['A'];rec[ix[1],ix[0]]=c['A'];assert torch.equal(rec,aa) and torch.equal(w,c['W']);ww,_=fit(rec,c['B'],st+4,alpha);assert float((ww-w).abs().max())<=1e-10;return p.stat().st_size

def evalcheck():
 y=torch.arange(60)%6;g=['toy'+str(i) for i in range(60)];names=['A','M','I','T','G_Matched','P_Matched','C_Matched'];s={(a,k):torch.zeros(60,k+4,dtype=torch.float64) for a in names for k in range(3)}
 for k in range(3):
  ix=torch.where(y<k+4)[0];s['A',k][ix,y[ix]]=1
 z=evaluate(s,y,g,[]);assert z['gates']['pass_all'] and z['mechanism_structured_orbit']['gatekeeping_pass'];s['A',0]=s['G_Matched',0].clone();assert not evaluate(s,y,g,[])['gates']['pass_all'];assert 'UNRESOLVED' in evaluate({},y,g,[])['decision'];json.dumps(z,allow_nan=False)
 f=dict(BA=.5,recall=[.5,2/3,.5,.5]);a=dict(BA=.5+1/12,recall=[.5,1.,.5,.5]);tab={c:{'0':dict(f)} for c in ARMS};tab['A']['0']=a;assert devgate(tab,0)['pass_all'];tab['A']['0']=f;assert not devgate(tab,0)['pass_all'];return dict(success_failure_missing_JSON=True,mechanism_three_gatekeeping=True,development_gate=True)

def preflight(cfg):
 out=Path(cfg['output']);s=synthetic();s.update(evalcheck());obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);tr=~obj['valid'];ids=torch.cat([torch.where(tr & (obj['y']==c))[0][:16] for c in range(4)]);assert len(ids)==64;m,*_=setup(cfg);index=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());pics={};err=0
 with torch.no_grad():
  for cell in index['chunks']:
   selected=[int(i) for i in ids if cell['first']<=int(i)<cell['first']+cell['rows']]
   if not selected:continue
   x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False)
   for start in sorted({(i-cell['first'])//32*32 for i in selected}):
    z=x[start:start+32];gg=features(m,z);actual=[i for i in selected if start<=i-cell['first']<start+len(z)]
    for i in actual:err=max(err,float((gg[i-cell['first']-start]-obj['g'][i]).abs().max()));pics[i]=x[i-cell['first']]
  p=torch.stack([pics[int(i)] for i in ids]);v=torch.stack([torch.cat([features(m,t) for t in xx.split(32)]) for xx in views(p)],1);p2=views(p)[1];vv=torch.stack([torch.cat([features(m,t) for t in xx.split(32)]) for xx in views(p2)],1)
 assert err<=1e-10;invariance=float((v.mean(1)-vv.mean(1)).abs().max());assert invariance<=1e-6;disp=float((v-v.mean(1,keepdim=True)).square().sum(-1).mean().sqrt());q=weights(obj['y'][ids],[obj['groups'][i] for i in ids]);mp=mapper(v,q);st,c,bar=stats(transform(v,mp),obj['y'][ids],q);st['A']=((st['A'][0]+st['A'][0].T)/2,st['A'][1]);w,_=fit(*st['A'],4,.01);size=artifact(cfg,out,'TEMPLATE',0,st['A'],w,mp,.01);assert float(torch.linalg.eigvalsh(c).min())>=-1e-8
 write(out/'aggregate.json',dict(pass_all=True,structural_sensitivity_pass=disp>1e-3,decision='ENGINEERING_PASS' if disp>1e-3 else 'STRUCTURAL_NO_ORIENTATION_SENSITIVITY',identity_replay_error=err,D4_average_permuted_error=invariance,orbit_RMS=disp,capacity_template_bytes=size,training_only_rows=64,actual_orbit_forward_images=1024,not_classification_result=True,method_success=False,**s))

def run(cfg,access):
 out=Path(cfg['output']);pre=json.loads(Path(cfg['preflight']).read_text());assert pre['pass_all'];assert pre['structural_sensitivity_pass'];m,*_=setup(cfg);ref=Path(cfg['reference']);maps={};state={};heads={};alphas={};fits=[];sizes={};devx={a:[] for a in ARMS};devy=[];checks=[];coverage=[];train_groups=set();val_groups=set();parity=[]
 def done(decision,stage,test=False,extra=None):
  payload=dict(decision=decision,stopped_stage=stage,development=checks,fits=fits,artifact_bytes=sizes,alphas=alphas,coverage=coverage,identity_replay_errors=parity,test_used=test,method_success=False);payload.update(extra or {});write(out/'aggregate.json',payload);write(out/'status.json',dict(state='COMPLETE',decision=decision,stage=stage))
 for stage in range(3):
  access['stage']=stage;obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);y=obj['y'];gg=obj['groups'];tr=~obj['valid'];va=obj['valid'];tg={gg[i] for i in torch.where(tr)[0].tolist()};vg={gg[i] for i in torch.where(va)[0].tolist()};assert not tg&(vg|val_groups) and not train_groups&vg;train_groups|=tg;val_groups|=vg;assert int(tr.sum())==[2555,1812,28][stage];q=weights(y[tr],[gg[i] for i in torch.where(tr)[0].tolist()]);v,err=extract(cfg,m,stage,obj,out);parity.append(err);coverage.append(dict(stage=stage,rows=len(y),training=int(tr.sum()),validation=int(va.sum()),logical_forward_images=len(y)*8));devy.append(y[va].clone());k=stage+4
  if stage==0:
   assert torch.bincount(y[va]).tolist()==[68,3,38,572];maps['F']=mapper(obj['g'][tr,None].double(),q);common=mapper(v[tr],q)
   for arm in ['M','I','A']:maps[arm]=common
   for arm in ['F','M','I','A']:write(out/(arm+'.TRANSFORM_SEAL.json'),dict(initial_training_rows=2555,future_frozen=True,test_used=False,width=1535))
  new,c,bar=stats(transform(v[tr],maps['A']),y[tr],q);fg=transform(obj['g'].double(),maps['F']);fx=torch.cat([fg,torch.ones(len(fg),1,dtype=fg.dtype)],1);new['F']=(fx[tr].T@(fx[tr]*q[:,None]),fx[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]));devx['F'].append(fx[va].clone());tb=transform(v.mean(1),maps['F']);devx['T'].append(torch.cat([tb[va],torch.ones(int(va.sum()),1,dtype=tb.dtype)],1));mb=transform(v.mean(1),maps['A']);mx=torch.cat([mb,torch.ones(len(mb),1,dtype=mb.dtype)],1)
  for arm in ['M','I','A']:devx[arm].append(mx[va].clone())
  for arm in ['F','M','I','A']:
   aa,bb=new[arm];aa=(aa+aa.T)/2;old=state.get(arm,(torch.zeros_like(aa),torch.zeros_like(bb)));state[arm]=(old[0]+aa,old[1]+bb)
   if stage==0:
    opts=[]
    for alpha in [.0001,.01,1.]:
     w,e=fit(*state[arm],k,alpha);z=devx[arm][-1]@w[:,:k];ba=metrics(z,y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA'];fits.append(dict(arm=arm,stage=stage,alpha=alpha,residual=e,status='CERTIFIED',initial_validation_BA=ba));opts.append((ba,alpha,w))
    _,alphas[arm],w=max(opts,key=lambda t:(t[0],t[1]));write(out/(arm+'.SELECTION_SEAL.json'),dict(alpha=alphas[arm],grid=[.0001,.01,1.],test_used=False))
    if arm=='F':assert alphas[arm]==.01 and abs(max(opts,key=lambda t:(t[0],t[1]))[0]-.7015464540619339)<1e-12
   else:w,e=fit(*state[arm],k,alphas[arm]);fits.append(dict(arm=arm,stage=stage,alpha=alphas[arm],residual=e,status='CERTIFIED'))
   heads[arm,stage]=w;sizes[f'{arm}.{stage}']=artifact(cfg,out,arm,stage,state[arm],w,maps[arm],alphas[arm])
  heads['T',stage]=heads['F',stage];yy=torch.cat(devy);sc={a:{s:torch.cat(devx[a])@heads[a,s][:,:s+4] for s in range(stage+1)} for a in ARMS};tab=table(sc,yy);gate=devgate(tab,stage);checks.append(dict(stage=stage,summary=tab,gate=gate));write(out/'DEVELOPMENT.json',dict(stages=checks,alphas=alphas,not_independent_confirmation=True));write(out/'partial.json',dict(fits=fits,artifact_bytes=sizes,coverage=coverage));del obj,v,y,gg,fg,fx,tb,mb,mx,new,c,bar
  if gate is not None and not gate['pass_all']:done('STOP_ORBIT_DEVELOPMENT_NO_TEST',stage);return
 write(out/'MODEL_SEAL.json',dict(new_heads=9,F_control_heads=3,fit_units=20,alphas=alphas,artifact_bytes=sizes,test_used=False));access.update(stage=3,sealed_test=True);prior=torch.load(cfg['inherited_predictions'],weights_only=False);y=prior['y'];groups=prior['groups'];bc=json.loads(Path(cfg['benchmark_config']).read_text());rows=json.loads((Path(bc['output'])/'ISIC.test.rows.private.json').read_text());assert torch.equal(y,torch.tensor([x['label'] for x in rows])) and groups==[x['identity_component'] for x in rows] and not set(groups)&(train_groups|val_groups);old=torch.load(Path(bc['output'])/'ISIC.test.features.private.pt',weights_only=False);g=torch.nn.functional.normalize(old['x'].double(),dim=1);v,err=extract(cfg,m,3,dict(y=y,g=g),out);coverage.append(dict(stage=3,rows=len(y),logical_forward_images=len(y)*8));scores=dict(prior['scores']);par=0
 for arm in ARMS:
  z=transform(g,maps['F']) if arm=='F' else transform(v.mean(1),maps['F'] if arm=='T' else maps[arm]);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)
  for st in range(3):
   val=(x@heads[arm,st])[:,:st+4]
   if arm=='F':par=max(par,float((val-scores['G_Matched',st]).abs().max()));assert par<=1e-8
   else:scores[arm,st]=val
 assert len(scores)==174;torch.save(dict(scores=scores,y=y,groups=groups),out/'predictions.private.pt');result=evaluate(scores,y,groups,[]);result.update(F_score_parity=par,test_identity_replay_error=err,historical_test_exposed=True);done(result.pop('decision'),2,True,result)
