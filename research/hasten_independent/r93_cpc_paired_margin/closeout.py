import campaign as c
import torch,json,time,math,csv
from pathlib import Path
c.guard(c.cfg);torch.set_grad_enabled(False);cp=time.process_time();wt=time.time();root=c.root
x=json.loads((root/'RESULTS.json').read_text());end=json.loads((root/'CONTROL_EXIT_4.json').read_text());seal=json.loads((root/'BUILDERS_SEALED.json').read_text());assert end['exit_code']==0 and seal=={'states':24,'optimizer_steps':6144,'VAL_TEST':0};assert len(x['rows'])==480 and x['counts']['evaluation_contexts']==120 and x['counts']['builder_optimizer_steps']==6144
for s in range(6):
 for arm in ['CONTROL','MARGIN']:
  for kind in ['main','held']:
   q=torch.load(root/f'builder_{arm}_{kind}_s{s}.private.pt',map_location='cpu',weights_only=False)
   def finite(v):
    if torch.is_tensor(v):assert torch.isfinite(v).all()
    elif isinstance(v,dict):
     for a in v.values():finite(a)
    elif isinstance(v,(list,tuple)):
     for a in v:finite(a)
   finite(q);C=sum(map(len,c.rcfg['stages'][:s+1]));assert q['classes_in_fit']==[j for j in range(C) if kind=='main' or j%3!=1]
key=lambda r:(r['stage'],r['support_slot'],r['rep'],r['arm'],r['kind']);old={key(r):r for r in x['rows']};assert len(old)==480
stats=[];classes=[];maxerr=0.;fw=0;scorecalls=0;scorerows=0;classifierrows=0
for s in range(6):
 d=c.load_stage(c.rcfg,s);C=d['C'];z,y,_=c.query(d,'F');tail,_=c.meta(c.rcfg,C,s);oldmask=torch.arange(C)<sum(map(len,c.rcfg['stages'][:s]));gmodels={}
 for arm in ['CONTROL','MARGIN']:
  for kind in ['main','held']:
   g=c.Builder();g.load_state_dict(torch.load(root/f'builder_{arm}_{kind}_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);g.eval();gmodels[arm,kind]=g
 classbank={}
 for slot in range(5):
  for rep in range(4):
   c.guard(c.cfg);a,pa,b,pb,cov=c.pair(d,s,50000+slot+5*rep,list(range(C)))
   for arm in ['CONTROL','MARGIN']:
    for kind in ['main','held']:
     g=gmodels[arm,kind];ca,_=g(a,pa);cb,_=g(b,pb);fw+=2;sa=c.score(z,ca);sb=c.score(z,cb);scorecalls+=2;scorerows+=6*len(y);mask=torch.ones(C,dtype=torch.bool) if kind=='main' else torch.arange(C)%3==1
     for k,alpha in [(0,'NOOP'),(1,'HALF'),(2,'FULL')]:
      recalls=[]
      for sc in [sa,sb]:rr,_=c.recall(sc[:,:,k],y,C);recalls.append(rr);classifierrows+=len(y)
      rr=(recalls[0]+recalls[1])/2;original=old[s,slot,rep,arm,kind]
      for metric,mm in [('macro',mask),('tail',mask&tail)]:maxerr=max(maxerr,abs(float(rr[mm].mean()*100)-original[alpha+'_'+metric]))
      for name,mm in [('all',mask),('old',mask&oldmask),('new',mask&~oldmask),('head',mask&~tail),('tail',mask&tail)]:
       if mm.any():stats.append({'stage':s,'support_slot':slot,'rep':rep,'arm':arm,'kind':kind,'alpha':alpha,'group':name,'macro_recall_pct':float(rr[mm].mean()*100),'classes':int(mm.sum())})
      for j in range(C):classbank.setdefault((arm,kind,alpha,j),[]).append(float(rr[j]*100))
 for (arm,kind,alpha,j),v in classbank.items():classes.append({'stage':s,'arm':arm,'kind':kind,'alpha':alpha,'class_index':j,'F_query_count':int((y==j).sum()),'context_side_exposures':40*int((y==j).sum()),'mean_recall_pct':math.fsum(v)/20,'is_held':j%3==1,'is_tail':bool(tail[j]),'is_new':not bool(oldmask[j])})
assert maxerr<1e-8 and fw==960 and scorecalls==960
rows=x['rows'];mean=lambda rr,k:math.fsum(a[k] for a in rr)/len(rr)
select=lambda arm,kind,diff=False:[r for r in rows if r['arm']==arm and r['kind']==kind and (not diff or r['support_pair_different'])]
ratio=mean(select('MARGIN','main',True),'margin_variance')/mean(select('CONTROL','main',True),'margin_variance');assert abs(ratio-x['variance_ratio'])<1e-12
deltas={kind:{m:mean(select('MARGIN',kind),'FULL_'+m)-mean(select('CONTROL',kind),'FULL_'+m) for m in ['macro','tail']} for kind in ['main','held']};harm=torch.load(root/'tiny_harms.private.pt',map_location='cpu',weights_only=False);assert len(harm)==360 and sum(harm)==x['tiny_new_harm_total']
checks={'stability_at_least_10pct':ratio<=.9,'main_global_macro_tail_nonnegative':all(v>=-1e-8 for v in deltas['main'].values()),'main_at_least_4stages_nonnegative':sum(all(v>=-1e-8 for v in dd.values()) for dd in x['stage_main_deltas'])>=4,'candidate_vs_NOOP':x['candidate_minus_own_NOOP_pp']['macro']>1e-8 and x['candidate_minus_own_NOOP_pp']['tail']>=-1e-8,'held_macro_tail_nonnegative':all(v>=-1e-8 for v in deltas['held'].values()),'tiny_no_new_harm':sum(harm)==0};assert all(checks.values())==x['mechanism_gate']
for name,rr in [('GROUP_CONTEXT_METRICS.csv',stats),('CLASS_STAGE_METRICS.csv',classes)]:
 with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
c.write(root/'VERIFICATION.json',{'passed':True,'formal_steps':6144,'states':24,'finite_model_and_optimizer_states':True,'held_fit_class_exclusion':True,'result_rows':480,'fixed_contexts':120,'metric_reconstruction_max_error_pp':maxerr,'mechanism_gate':False,'gate_checks':checks,'saved_only_builder_forwards':fw,'saved_only_score_bank_calls':scorecalls,'saved_only_candidate_score_rows':scorerows,'saved_only_classifier_rows':classifierrows,'optimizer':0,'encoder':0,'VAL_TEST':0,'CPU_seconds':time.process_time()-cp,'wall_seconds':time.time()-wt})
c.write(root/'CLOSEOUT_STATUS.json',{'phase':'SAVED_VERIFICATION_COMPLETE','mechanism_gate':False,'optimizer':0,'encoder':0,'VAL_TEST':0})
