import os,sys,json,time,traceback,copy,math
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);sys.path.insert(0,cfg['r91'])
from common import write,save,guard
from core import Builder,state,unit,score
from binding import load_stage,query,meta
from train_build import loss_build
rcfg=json.loads((Path(cfg['r91'])/'config.private.json').read_text());torch.set_num_threads(4)
counter={'builder_optimizer_steps':0,'pair_backward':0,'support_episode':0,'query_training_rows':0,'engineering_pair_backward':0,'evaluation_contexts':0,'evaluation_builder_forward':0,'encoder':0,'actor_optimizer':0,'VAL_TEST':0};cpu=time.process_time();wall=time.time()

def pair(d,s,idx,classes):
 cap=[1,2,4,8,None][idx%5];sa=[];sb=[];coverage=[]
 for c in classes:
  pool=sorted(i for i in d['roles']['S'] if int(d['y'][d['index'][i]])==c);n=len(pool);m=n if cap is None else min(cap,n);a=(74002+100003*s+idx//5)%n;b=(a+max(1,m))%n if m<n else a
  ia=[pool[(a+j)%n] for j in range(m)];ib=[pool[(b+j)%n] for j in range(m)]
  sa.append(d['z'][[d['index'][i] for i in ia]]);sb.append(d['z'][[d['index'][i] for i in ib]]);coverage.append({'class':c,'m':m,'n':n,'overlap':len(set(ia)&set(ib)),'different':ia!=ib})
 sta,pa=state(sa,classes,s,rcfg['train_counts'],d['birth']);stb,pb=state(sb,classes,s,rcfg['train_counts'],d['birth']);return sta,pa,stb,pb,coverage

def consistency(a,b,y):
 # Already divided by original score scale20: inputs here are cosine scores.
 delta=a-b;diff=delta.gather(1,y[:,None])-delta;mask=~F.one_hot(y,a.shape[1]).bool();row=(diff.square()*mask).sum(1)/(a.shape[1]-1)
 return torch.stack([row[y==c].mean() for c in range(a.shape[1])]).mean()

def objective(g,d,s,idx,classes,z,y,lam):
 a,pa,b,pb,cov=pair(d,s,idx,classes);ca,_=g(a,pa);cb,_=g(b,pb)
 def supervised(c):
  ce=F.cross_entropy(20*z@c[:,2].T,y,reduction='none');ce=torch.stack([ce[y==j].mean() for j in range(len(classes))]).mean();target=torch.stack([unit(z[y==j].mean(0)) for j in range(len(classes))]);return ce+.1*(1-(c[:,2]*target).sum(1)).mean()
 loss=(supervised(ca)+supervised(cb))/2;mc=consistency(z@ca[:,2].T,z@cb[:,2].T,y)
 return loss+lam*mc,float(mc.detach()),cov

def init():
 torch.manual_seed(74002);g=Builder();return g,torch.optim.AdamW(g.parameters(),lr=3e-4,weight_decay=1e-4)

def software():
 torch.manual_seed(93);a=torch.randn(7,3,dtype=torch.float64,requires_grad=True);b=torch.randn(7,3,dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,2,0,1,2,0]);v=consistency(a,b,y);shift=torch.randn(7,1,dtype=torch.float64)
 assert abs(float(v-consistency(a+shift,b,y)))<1e-12;assert float(consistency(a,a,y))==0
 perm=torch.tensor([2,0,1]);inv=torch.argsort(perm);assert abs(float(v-consistency(a[:,perm],b[:,perm],inv[y])))<1e-12
 v.backward();assert a.grad.norm()>0 and b.grad.norm()>0;assert torch.isfinite(a.grad).all()
 write(root/'SOFTWARE.json',{'passed':True,'common_shift_invariance':True,'identical_pair_zero':True,'class_permutation':True,'both_sides_gradient':True,'optimizer':0,'encoder':0,'VAL_TEST':0})

def recall(sc,y,C):
 pred=sc.argmax(1);return torch.stack([(pred[y==c]==c).double().mean() for c in range(C)]),pred

def main():
 software();d=load_stage(rcfg,0);classes=list(range(d['C']));z,y,_=query(d,'W',classes);g,opt=init();before={k:v.clone() for k,v in g.state_dict().items()};times=[]
 for repeat in range(2):
  t=time.time();g.zero_grad();loss,mc,cov=objective(g,d,0,2,classes,z,y,1.);loss.backward();elapsed=time.time()-t;times.append(elapsed);counter['engineering_pair_backward']+=1;assert torch.isfinite(loss);assert g.net[-1].weight.grad.norm()>0
 assert all(torch.equal(v,before[k]) for k,v in g.state_dict().items());g.load_state_dict(before,strict=True);g.zero_grad()
 # Fixed second pass; conservative worst-size projection and safety4, plus analysis reserve.
 factor=(23/d['C'])**2*(1713/len(y));projected=times[1]*6144*4*max(1,factor)*4+3600;remaining=cfg['deadline']-time.time();passed=projected<remaining-180
 write(root/'NATIVE_BUDGET.json',{'passed':passed,'pair_times_seconds':times,'fixed_second_pass':True,'worst_size_factor':factor,'safety_factor':4,'projected_seconds':projected,'remaining_seconds':remaining,'engineering_optimizer':0,'encoder':0,'TRAIN_pair_gradient_finite':True,'strict_restore':True,'counts':counter.copy()})
 if not passed:write(root/'STATUS.json',{'phase':'NOT_ADMITTED_BUDGET','counts':counter});return
 models={};opts={}
 for arm in ['CONTROL','MARGIN']:
  for kind in ['main','held']:models[arm,kind],opts[arm,kind]=init()
 coverage=[]
 for s in range(6):
  d=load_stage(rcfg,s)
  for arm in ['CONTROL','MARGIN']:
   for kind in ['main','held']:
    g=models[arm,kind];opt=opts[arm,kind];classes=[c for c in range(d['C']) if kind=='main' or c%3!=1];z,y,_=query(d,'W',classes)
    for update in range(256):
     guard(cfg);opt.zero_grad();tot=0.;mcs=0.
     for ep in range(4):
      loss,mc,cov=objective(g,d,s,4*update+ep,classes,z,y,float(arm=='MARGIN'));assert torch.isfinite(loss);(loss/4).backward();tot+=float(loss.detach())/4;mcs+=mc/4;counter['pair_backward']+=1;counter['support_episode']+=2;counter['query_training_rows']+=2*len(y)
      coverage.append({'stage':s,'arm':arm,'kind':kind,'pair':4*update+ep,'coverage':cov})
     opt.step();counter['builder_optimizer_steps']+=1
     if update%16==15:write(root/'STATUS.json',{'phase':'BUILD','stage':s,'arm':arm,'kind':kind,'update_complete':update+1,'loss':tot,'margin_consistency':mcs,'counts':counter})
    save(root/f'builder_{arm}_{kind}_s{s}.private.pt',{'model':g.state_dict(),'optimizer':opt.state_dict(),'counts':counter.copy(),'classes_in_fit':classes})
  save(root/'pair_coverage.private.pt',coverage)
 assert counter['builder_optimizer_steps']==6144
 write(root/'BUILDERS_SEALED.json',{'states':24,'optimizer_steps':6144,'VAL_TEST':0})
 torch.set_grad_enabled(False);rows=[];harms=[]
 for s in range(6):
  d=load_stage(rcfg,s);C=d['C'];qz,qy,_=query(d,'F');tail,protected=meta(rcfg,C,s);held=torch.arange(C)%3==1
  for slot in range(5):
   for rep in range(4):
    guard(cfg);idx=50000+slot+5*rep;a,pa,b,pb,cov=pair(d,s,idx,list(range(C)));different=any(t['different'] for t in cov)
    preds={}
    for arm in ['CONTROL','MARGIN']:
     for kind in ['main','held']:
      g=Builder();g.load_state_dict(torch.load(root/f'builder_{arm}_{kind}_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);g.eval();ca,ra=g(a,pa);cb,rb=g(b,pb);counter['evaluation_builder_forward']+=2
      sa=score(qz,ca);sb=score(qz,cb);mc=float(consistency(sa[:,:,2]/20,sb[:,:,2]/20,qy));mask=torch.ones(C,dtype=torch.bool) if kind=='main' else held;tm=tail&mask
      item={'stage':s,'support_slot':slot,'rep':rep,'arm':arm,'kind':kind,'support_pair_different':different,'margin_variance':mc,'residual_norm':float((ra.norm(dim=1).mean()+rb.norm(dim=1).mean())/2)}
      for k,name in [(0,'NOOP'),(1,'HALF'),(2,'FULL')]:
       rs=[];pp=[]
       for scores in [sa,sb]:r,pred=recall(scores[:,:,k],qy,C);rs.append(r);pp.append(pred)
       rr=(rs[0]+rs[1])/2;item[name+'_macro']=float(rr[mask].mean()*100);item[name+'_tail']=float(rr[tm].mean()*100)
       if k==2:preds[arm,kind]=pp
       if k==0:np=pp
       if k==2:item['NOOP_prediction_change_rate']=sum(float((p!=n).double().mean()) for p,n in zip(pp,np))/2
      rows.append(item)
    for c in [1,2,12]:
     ix=qy==c;control=preds['CONTROL','main'];candidate=preds['MARGIN','main'];harms.append(sum(int(((cp[ix]==c)&(mp[ix]!=c)).sum()) for cp,mp in zip(control,candidate)))
    counter['evaluation_contexts']+=1;write(root/'STATUS.json',{'phase':'EVALUATE_TRAIN','completed_contexts':counter['evaluation_contexts'],'counts':counter})
 def mean(ss,k):return sum(r[k] for r in ss)/len(ss)
 def select(arm,kind,diff=False):return [r for r in rows if r['arm']==arm and r['kind']==kind and (not diff or r['support_pair_different'])]
 control=select('CONTROL','main',True);candidate=select('MARGIN','main',True);ratio=mean(candidate,'margin_variance')/mean(control,'margin_variance') if mean(control,'margin_variance')>1e-12 else None
 diffs={kind:{m:mean(select('MARGIN',kind),'FULL_'+m)-mean(select('CONTROL',kind),'FULL_'+m) for m in ['macro','tail']} for kind in ['main','held']}
 staged=[{m:mean([r for r in select('MARGIN','main') if r['stage']==s],'FULL_'+m)-mean([r for r in select('CONTROL','main') if r['stage']==s],'FULL_'+m) for m in ['macro','tail']} for s in range(6)]
 own={m:mean(select('MARGIN','main'),'FULL_'+m)-mean(select('MARGIN','main'),'NOOP_'+m) for m in ['macro','tail']}
 gate=ratio is not None and ratio<=.9 and all(v>=-1e-8 for dd in diffs.values() for v in dd.values()) and sum(all(v>=-1e-8 for v in dd.values()) for dd in staged)>=4 and own['macro']>1e-8 and own['tail']>=-1e-8 and sum(harms)==0
 save(root/'tiny_harms.private.pt',harms)
 write(root/'RESULTS.json',{'phase':'COMPLETE','mechanism_gate':gate,'variance_ratio':ratio,'candidate_minus_control_pp':diffs,'candidate_minus_own_NOOP_pp':own,'stage_main_deltas':staged,'tiny_new_harm_total':sum(harms),'rows':rows,'counts':counter,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'limitations':'TRAIN development diagnosis; no actor or VAL/TEST; FULL S differs from all-TRAIN deployment; no originality or GRPO efficacy claim.'})
 write(root/'STATUS.json',{'phase':'COMPLETE','mechanism_gate':gate,'counts':counter})
try:main()
except BaseException:
 (root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
