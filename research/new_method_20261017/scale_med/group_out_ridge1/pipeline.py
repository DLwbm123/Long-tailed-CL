import json,os,sys
from pathlib import Path
import torch,numpy as np
from train_base import setup
from binding import native
from core_depth import PERM
from core_attribution import weights
from solver import mapper,transform
from train import write
from evaluation import evaluate,metrics
from space import check_space
ARMS=['F','E','I','O'];BUDGET=19058800

def fit(a,b,k,alpha):
 h=a/k;h=h.clone();h.diagonal()[:-1].add_(alpha);w=torch.linalg.solve(h,b/k);e=float((h@w-b/k).abs().max());assert e<=1e-7 and torch.isfinite(w).all();return w,e

def points(scores,y):
 result={}
 for arm in scores:
  result[arm]={}
  for st,v in scores[arm].items():
   mask=y<st+4;yy=y[mask];pred=v[mask].argmax(1);rec=[float((pred[yy==c]==c).double().mean()) for c in range(st+4)];result[arm][str(st)]=dict(BA=sum(rec)/(st+4),recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(st+4)],counts=[int((yy==c).sum()) for c in range(st+4)])
 return result

def initial_gate(tab):
 a,b=tab['O']['0'],tab['F']['0'];d=np.array(a['recall'])-b['recall'];return dict(BA_gain_pp=100*(a['BA']-b['BA']),class1_gain_pp=100*d[1],other_class_gains_pp=(100*d[[0,2,3]]).tolist(),pass_all=bool(a['BA']-b['BA']>=.01-1e-12 and d[1]>=.05-1e-12 and min(d[[0,2,3]])>=-.01-1e-12))
def future_gate(tab):
 o,f,i=tab['O'],tab['F'],tab['I'];J=lambda t:50*(t['1']['BA']+t['2']['BA']);classes=[100*(o[str(s)]['recall'][1]-f[str(s)]['recall'][1]) for s in range(3)];birth=[100*(o[str(s)]['recall'][c]-f[str(s)]['recall'][c]) for s,c in [(1,4),(2,4),(2,5)]];old=[100*np.mean(np.array(o[str(s)]['recall'][:4])-f[str(s)]['recall'][:4]) for s in [1,2]]+[100*np.mean(np.array(o['2']['recall'][:5])-f['2']['recall'][:5])];ba=[100*(o[str(s)]['BA']-f[str(s)]['BA']) for s in [1,2]];class0=[100*(o[str(s)]['recall'][0]-f[str(s)]['recall'][0]) for s in range(3)];r=dict(J_O_minus_F_pp=J(o)-J(f),J_O_minus_I_pp=J(o)-J(i),class1_gains_pp=classes,birth_retired_gains_pp=birth,old_macro_gains_pp=old,future_BA_gains_pp=ba,class0_gains_pp=class0);r['pass_all']=bool(r['J_O_minus_F_pp']>=1-1e-12 and r['J_O_minus_I_pp']>=.5-1e-12 and min(classes)>=5-1e-12 and min(birth)>=-1e-12 and min(old)>=-1-1e-12 and min(ba)>=-1e-12 and min(class0)>=-1-1e-12);return r

def synthetic():
 y=torch.arange(60)%6;groups=['toy'+str(i) for i in range(60)];names=['G_Matched','P_Matched','C_Matched','E','I','O'];scores={(n,s):torch.zeros(60,s+4,dtype=torch.float64) for n in names for s in range(3)}
 for s in range(3):
  mask=y<s+4;scores['O',s][torch.arange(60)[mask],y[mask]]=1
 success=evaluate(scores,y,groups,[]);assert success['gates']['pass_all'] and success['mechanism_O_minus_I']['gatekeeping_pass'];scores['O',0]=scores['G_Matched',0].clone();fail=evaluate(scores,y,groups,[]);assert not fail['gates']['pass_all'];assert evaluate({},y,groups,[])['decision']=='UNRESOLVED_GROUP_OUT_RIDGE1';json.dumps(success,default=lambda x:x.item() if isinstance(x,np.generic) else x,allow_nan=False)
 base=dict(BA=.5,recall=[.5,2/3,.5,.5]);good=dict(BA=.5+1/12,recall=[.5,1.,.5,.5]);dev={'F':{'0':base},'O':{'0':good}};assert initial_gate(dev)['pass_all'];bad=dict(BA=good['BA'],recall=[.5-1/68,1.,.5,.5]);dev['O']['0']=bad;assert not initial_gate(dev)['pass_all']
 return dict(development_discrete_gate_success_failure=True,full_six_comparisons_and_mechanism=True,rescue_harm_identity=True,success_failure_missing_JSON=True)

def run(cfg,access):
 out=Path(cfg['output']);ref=Path(cfg['reference']);matched=Path(cfg['matched']);training=Path(cfg['training_output']);assert json.loads((training/'ENCODER_SEAL.json').read_text())['future_frozen'];maps={};state={};heads={};alphas={};fits=[];sizes={};devx={a:[] for a in ARMS};devy=[];checks=[];counts=[];forward={};stage=-1;train_groups=set();val_groups=set()
 def extract(arm,st,expected):
  m,params,bkeys,ckeys,hs,pw=setup(cfg);values=[];bias=torch.load(training/(arm+'.encoder.private.pt'),weights_only=False)['bias_overrides'];over=native(bias)
  with torch.no_grad():
   for k in bkeys:params[k].copy_(over[k].cuda())
   index=json.loads((matched/f'pixels_{st}/INDEX.json').read_text());assert index['rows']==expected
   for cell in index['chunks']:
    x=torch.load(matched/f'pixels_{st}'/cell['file'],weights_only=False)
    for z in x.split(32):values.append(torch.nn.functional.normalize(m(z.cuda()).cpu().double(),dim=1))
  del m,params,hs;torch.cuda.empty_cache();forward[arm]=forward.get(arm,0)+expected;g=torch.cat(values);assert torch.isfinite(g).all();return g,bias
 def save():write(out/'partial.json',dict(fits=fits,counts=counts,artifact_bytes=sizes,feature_rows=forward))
 for stage in range(3):
  access['stage']=stage;write(out/'status.json',dict(state='RUNNING',phase='stage_features_and_statistics',stage=stage));obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);y=obj['y'];gg=obj['groups'];tr=~obj['valid'];va=obj['valid'];tg={gg[i] for i in torch.where(tr)[0].tolist()};vg={gg[i] for i in torch.where(va)[0].tolist()};assert not tg&(vg|val_groups) and not train_groups&vg;train_groups|=tg;val_groups|=vg;q=weights(y[tr],[gg[i] for i in torch.where(tr)[0].tolist()]);assert tr.sum()==[2555,1812,28][stage];counts.append(dict(stage=stage,training=int(tr.sum()),validation=int(va.sum())));devy.append(y[va]);k=stage+4
  if stage==0:assert torch.bincount(y[va]).tolist()==[68,3,38,572]
  for arm in ARMS:
   check_space(cfg['window_root']);g,bias=(obj['g'].double(),None) if arm=='F' else extract(arm,stage,len(y))
   if stage==0:
    maps[arm]=mapper(g[tr],q);write(out/(arm+'.TRANSFORM_SEAL.json'),dict(training_rows=2555,width=1535,initial_only=True,test_used=False))
   z=transform(g,maps[arm]);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);aa=x[tr].T@(x[tr]*q[:,None]);aa=(aa+aa.T)/2;bb=x[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);old=state.get(arm,(torch.zeros_like(aa),torch.zeros_like(bb)));state[arm]=(old[0]+aa,old[1]+bb);devx[arm].append(x[va].clone())
   if stage==0:
    options=[]
    for alpha in [.0001,.01,1.]:
     w,e=fit(*state[arm],k,alpha);ba=metrics(x[va]@w[:,:k],y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA'];fits.append(dict(arm=arm,stage=stage,alpha=alpha,residual=e,status='CERTIFIED',initial_validation_BA=ba));options.append((ba,alpha,w))
    _,alphas[arm],w=max(options,key=lambda t:(t[0],t[1]));write(out/(arm+'.SELECTION_SEAL.json'),dict(alpha=alphas[arm],grid=[.0001,.01,1.],source='own_initial_G_validation_681',test_used=False))
    if arm=='F':assert alphas[arm]==.01
   else:w,e=fit(*state[arm],k,alphas[arm]);fits.append(dict(arm=arm,stage=stage,alpha=alphas[arm],residual=e,status='CERTIFIED'))
   heads[arm,stage]=w
   if arm!='F':
    aa,bb=state[arm];ix=torch.triu_indices(len(aa),len(aa));artifact=dict(A=aa[ix.unbind()].contiguous(),B=bb.contiguous(),W=w.contiguous(),mapper=maps[arm],PERM=PERM,bias_overrides=bias,bias_representation='absolute_overrides',alpha=alphas[arm],classes=k,source_commit=cfg['source_commit']);file=out/f'{arm}.{stage}.state.private.pt';check_space(cfg['window_root'],reserve=24*2**20);torch.save(artifact,file);sizes[f'{arm}.{stage}']=file.stat().st_size;assert sizes[f'{arm}.{stage}']<=BUDGET;loaded=torch.load(file,weights_only=False);assert torch.equal(loaded['W'],w) and all(torch.equal(loaded['bias_overrides'][key],v) for key,v in bias.items());rec=torch.zeros_like(aa);rec[ix.unbind()]=loaded['A'];rec[ix[1],ix[0]]=loaded['A'];assert torch.equal(rec,aa);ww,_=fit(rec,loaded['B'],k,alphas[arm]);assert float((ww-w).abs().max())<=1e-10
   del g,x,z
  yy=torch.cat(devy);sc={a:{s:torch.cat(devx[a])@heads[a,s][:,:s+4] for s in range(stage+1)} for a in ARMS};tab=points(sc,yy);gate=initial_gate(tab) if stage==0 else (future_gate(tab) if stage==2 else None);checks.append(dict(stage=stage,summary=tab,gate=gate));save();write(out/'DEVELOPMENT.json',dict(stages=checks,alpha=alphas,historical_development_exposed=True,not_independent_confirmation=True))
  del obj,y,gg
  if gate is not None and not gate['pass_all']:
   write(out/'aggregate.json',dict(decision='STOP_DEVELOPMENT_GATE_NO_TEST',stopped_stage=stage,development=checks,fits=fits,artifact_bytes=sizes,alphas=alphas,feature_rows=forward,test_used=False,method_success=False));return
 write(out/'MODEL_SEAL.json',dict(new_heads=9,control_heads=3,new_fit_units=15,total_with_F_replay=20,alphas=alphas,artifact_bytes=sizes,test_used=False,source_commit=cfg['source_commit']));access['sealed_test']=True;access['stage']=3
 prior=torch.load(cfg['inherited_predictions'],weights_only=False);y=prior['y'];groups=prior['groups'];bc=json.loads(Path(cfg['benchmark_config']).read_text());rows=json.loads((Path(bc['output'])/'ISIC.test.rows.private.json').read_text());assert torch.equal(y,torch.tensor([r['label'] for r in rows])) and groups==[r['identity_component'] for r in rows] and not set(groups)&(train_groups|val_groups);test=torch.load(Path(bc['output'])/'ISIC.test.features.private.pt',weights_only=False);g=torch.nn.functional.normalize(test['x'].double(),dim=1);scores=dict(prior['scores']);parity=0
 for arm in ARMS:
  gg,bias=(g,None) if arm=='F' else extract(arm,3,len(y));z=transform(gg,maps[arm]);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)
  for s in range(3):
   v=(x@heads[arm,s])[:,:s+4]
   if arm=='F':parity=max(parity,float((v-scores['G_Matched',s]).abs().max()));assert parity<=1e-8
   else:scores[arm,s]=v
 assert len(scores)==171;torch.save(dict(scores=scores,y=y,groups=groups),out/'predictions.private.pt');result=evaluate(scores,y,groups,[]);result.update(source_commit=cfg['source_commit'],development=checks,fits=fits,artifact_bytes=sizes,alphas=alphas,feature_rows=forward,F_score_parity=parity,historical_test_exposed=True,patient_identity='UNKNOWN');write(out/'aggregate.json',result)
