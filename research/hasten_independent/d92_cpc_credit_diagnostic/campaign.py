import os,sys,json,time,itertools,resource,traceback
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);src=Path(cfg['r91']);sys.path.insert(0,str(src))
from common import write,guard
import train_build as tb
import binding,core
cpu=time.process_time();wall=time.time();torch.set_num_threads(4);torch.set_grad_enabled(False)
counts={'builder_forward':0,'candidate_score_bank_calls':0,'candidate_score_rows':0,'utility_calls':0,'utility_action_rows':0,'reference_classifier_rows':0,'optimizer':0,'encoder':0,'VAL_TEST':0}
origscore=tb.score;origutil=tb.utility;origref=tb.reference
# Imported functions resolve these instrumented aliases in table().
def score(z,p):
 counts['candidate_score_bank_calls']+=1;counts['candidate_score_rows']+=len(z)*3;return origscore(z,p)
def util(sc,y,a,*args):
 counts['utility_calls']+=1;counts['utility_action_rows']+=len(y)*(1 if a.ndim==1 else len(a));return origutil(sc,y,a,*args)
def ref(c,d,s,ids,sc):
 if s:counts['reference_classifier_rows']+=sum(int(d['y'][d['index'][i]])<sum(map(len,c['stages'][:s])) for i in ids)
 return origref(c,d,s,ids,sc)
tb.score=score;tb.reference=ref
acts=torch.tensor(list(itertools.product(range(3),repeat=4)))
def gradient(v):
 q=torch.stack([torch.stack([v[acts[:,c]==k].mean() for k in range(3)]) for c in range(4)])
 return (q-q.mean(1,keepdim=True))/3
v=torch.arange(81,dtype=torch.float64);g=gradient(v);assert float(g.sum(1).abs().max())<1e-12
# Conditional averaging must recover exact uniform value for every class.
assert all(abs(float(torch.stack([v[acts[:,c]==k].mean() for k in range(3)]).mean()-v.mean()))<1e-12 for c in range(4))
write(out/'SOFTWARE.json',{'passed':True,'scope':'new 81-action conditional centering/mean identity','optimizer':0,'real_data':0})
rcfg=json.loads((src/'config.private.json').read_text());base=json.loads((src/'PRECHECK.json').read_text());rows=[];maxerr=0.;qprivate=[]
def metrics(sc,y,a,t):return util(sc,y,a,t['tail'],t['protected'],t['reference'])
def mdelta(rec,base,mask):return float((rec-base)[mask].mean()*100)
try:
 for s in range(6):
  guard(cfg);d=tb.load_stage(rcfg,s);C=d['C'];models={}
  for name in ['main','held']:
   b=core.Builder();b.load_state_dict(torch.load(src/f'builder_{name}_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);b.eval();models[name]=b
  for idx in range(5):
   guard(cfg);t=tb.table(rcfg,d,s,idx,models['main']);h=tb.table(rcfg,d,s,idx,models['held']);counts['builder_forward']+=2
   zero=torch.zeros(C,dtype=torch.long);full=torch.full((C,),2,dtype=torch.long);held=torch.arange(C)%3==1;ht=held&t['tail'];four=torch.stack([zero,torch.where(held,full,zero),torch.where(held,zero,full),full]);hv,hr,_=metrics(h['scores'],h['y'],four,h)
   r={'stage':s,'support_slot':idx};effects={}
   for label,mask in [('macro',held),('tail',ht)]:
    selfd=mdelta(hr[1],hr[0],mask);comp=mdelta(hr[2],hr[0],mask);interaction=float((hr[3]-hr[1]-hr[2]+hr[0])[mask].mean()*100);total=mdelta(hr[3],hr[0],mask)
    assert abs(selfd+comp+interaction-total)<1e-8
    r.update({f'held_{label}_{k}_pp':v for k,v in [('self',selfd),('competitor',comp),('interaction',interaction),('total',total)]})
   block=zero.repeat(81,1);block[:,:4]=acts
   bv,br,_=util(t['B_scores'],t['B_y'],block,t['tail'],t['protected'],t['B_reference']);fv,fr,_=metrics(t['scores'],t['y'],block,t)
   chosen=int(bv[:,2].argmax());oracle=int(fv[:,2].argmax());delta=fr[chosen]-fr.mean(0)
   r['block_macro_pp']=float(delta.mean()*100);r['block_tail_pp']=float(delta[t['tail']].mean()*100)
   for name,values,recs in [('B',bv,br),('F',fv,fr)]:
    r[name+'_selected_D_delta']=float(values[chosen,1]-values[:,1].mean());r[name+'_selected_U_delta']=float(values[chosen,2]-values[:,2].mean())
    for tag,aidx in [('fixed0',0),('fixedhalf',40),('fixedfull',80)]:
     r[f'{name}_{tag}_U_delta']=float(values[aidx,2]-values[:,2].mean())
   headroom=float(fv[oracle,2]-fv[:,2].mean());regret=float(fv[oracle,2]-fv[chosen,2]);assert abs(headroom-regret-r['F_selected_U_delta'])<1e-10
   r.update({'F_U_headroom':headroom,'F_U_regret':regret})
   gb=gradient(bv[:,2]);gf=gradient(fv[:,2]);nb=float(gb.norm());nf=float(gf.norm());dot=float((gb*gf).sum());r.update({'gradient_B_norm':nb,'gradient_F_norm':nf,'gradient_dot':dot,'gradient_cosine':dot/(nb*nf) if nb*nf>1e-24 else None})
   for k,col in [('benefit',0),('risk',1)]:
    bg=gradient(bv[:,col]);fg=gradient(fv[:,col]);r[k+'_gradient_dot']=float((bg*fg).sum());r[k+'_gradient_B_norm']=float(bg.norm());r[k+'_gradient_F_norm']=float(fg.norm())
   original=base['rows'][s*5+idx]
   for key,value in [('held_macro_pp',r['held_macro_total_pp']),('held_tail_pp',r['held_tail_total_pp']),('block_macro_pp',r['block_macro_pp']),('block_tail_pp',r['block_tail_pp'])]:maxerr=max(maxerr,abs(value-original[key]))
   assert maxerr<1e-8
   qprivate.append({'stage':s,'slot':idx,'B_values':bv,'F_values':fv,'B_gradient':gb,'F_gradient':gf})
   rows.append(r);write(out/'STATUS.json',{'phase':'TRAIN_DIAGNOSTIC','completed_contexts':len(rows),'counts':counts})
  del d,models
 torch.save(qprivate,out/'tables.private.pt')
 write(out/'RESULTS.json',{'phase':'COMPLETE','rows':rows,'counts':counts,'reconstruction_max_error_pp':maxerr,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'scope':'Frozen TRAIN diagnostic, no optimization/encoder/VAL. Oracle is not deployment. Local logit gradient is not shared actor gradient.'})
 write(out/'STATUS.json',{'phase':'COMPLETE','completed_contexts':30,'counts':counts})
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
