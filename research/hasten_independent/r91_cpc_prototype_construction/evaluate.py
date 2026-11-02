import csv,json,torch,time
from pathlib import Path
from core import *
from native import assets,pixels
from common import *

def metrics(rec,cnt,tail):return {'micro':float((rec*cnt).sum()/cnt.sum()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100),'class_counts':cnt.tolist(),'class_correct':(rec*cnt).tolist(),'class_recall_percent':(rec*100).tolist()}
def run(cfg,counter):
 root=Path(cfg['output']);seal=json.loads((root/'ALL_STATES_SEALED.json').read_text());assert seal['VAL_reads']==0 and seal['actor_states']==108
 with open(Path(cfg['manifest'])/'val.csv') as f:rows=list(csv.DictReader(f))
 assert len(rows)==1698 and all(x['split']=='val' for x in rows);y=torch.tensor([cfg['order'].index(int(x['original_label'])) for x in rows]);visual,tail,tf,ops=assets(cfg);adapt,load,prefix=ops;tokens=[];wall=time.time();cpu=time.process_time()
 with torch.no_grad():
  for begin in range(0,len(rows),32):
   guard(cfg);x=pixels(cfg,rows,list(range(begin,min(begin+32,len(rows)))),tf);tokens.append(prefix(visual,x).cpu());counter['VAL_prefix_encoder_rows']+=len(x)
  del visual;torch.cuda.empty_cache();adapt(tail);tail.eval().requires_grad_(False);records=[]
  for s in range(6):
   head=torch.load(Path(cfg['source_root'])/f'r67_clip_grpo/SFT/HK/74002/head{s}.private.pt',map_location='cpu',weights_only=False);load(tail,head['delta']);vectors=[]
   for tok in tokens:
    guard(cfg);vectors.append(tail(tok.cuda()).cpu());counter['VAL_tail_encoder_rows']+=len(tok)
   z=torch.cat(vectors);sel=y<head['seen'];yy=y[sel];C=head['seen'];tailmask=torch.tensor([n<250 for n in cfg['train_counts'][:C]]);cnt=torch.bincount(yy,minlength=C).to(DT);sc=None
   for step in [64,128,256]:
    deploy=torch.load(root/f'deploy_s{s}_u{step}.private.pt',map_location='cpu',weights_only=False)
    if sc is None:sc=score(z[sel].to(DT),deploy['proto']);save(root/f'VAL_scores_s{s}.private.pt',{'scores':sc,'labels':yy,'ids':torch.where(sel)[0],'S_dep_ids':deploy['S_dep_ids']});counter['VAL_classifier_score_rows']+=len(yy)*3
    assert torch.equal(deploy['proto'][:,0],head['prototypes'].to(DT))
    for arm,prob in deploy['distributions'].items():
     _,rec=exact_B(prob,wins(sc,yy),yy,tailmask);a=prob.argmax(1);cc,nn=counts_correct(sc[:,torch.arange(C),a],yy,C);modal=cc/nn
     item={'stage':s,'updates':step,'arm':arm,'expected':metrics(rec,cnt,tailmask),'modal':metrics(modal,cnt,tailmask),'zero_action_class_fraction':float((a==0).to(DT).mean())};records.append(item)
   write(root/'STATUS.json',{'phase':'VAL','stage':s,'counts':counter})
 write(root/'METRICS.json',records);write(root/'EVAL_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'counts':counter,'TEST_reads':0});return records

def analyze(cfg):
 root=Path(cfg['output']);rows=json.loads((root/'METRICS.json').read_text());r={(v['stage'],v['arm']):v for v in rows if v['updates']==256};arms=list(r[(0,'N0')]) if False else list(r0['arm'] for r0 in rows if r0['stage']==0 and r0['updates']==256)
 summary={a:{k:sum(r[s,a]['modal'][k] for s in range(6))/6 for k in ['micro','macro','tail']} for a in arms};final={a:r[5,a] for a in arms};delta={a:{'mean_macro':summary['M']['macro']-summary[a]['macro'],'mean_tail':summary['M']['tail']-summary[a]['tail'],'final_macro':final['M']['modal']['macro']-final[a]['modal']['macro'],'final_tail':final['M']['modal']['tail']-final[a]['modal']['tail']} for a in arms if a!='M'}
 tiny=[1,2,12];base=delta['N0'];samecase=[]
 for s in range(6):
  v=torch.load(root/f'VAL_scores_s{s}.private.pt',map_location='cpu',weights_only=False);d=torch.load(root/f'deploy_s{s}_u256.private.pt',map_location='cpu',weights_only=False);C=v['scores'].shape[1];pred={}
  for a in ['M','N0']:
   chosen=d['distributions'][a].argmax(1);pred[a]=v['scores'][:,torch.arange(C),chosen].argmax(1)
  yy=v['labels'];cnt=torch.bincount(yy,minlength=C);gain=(pred['M']==yy)&(pred['N0']!=yy);harm=(pred['M']!=yy)&(pred['N0']==yy);samecase.append({'stage':s,'saved_correct_rescued':torch.bincount(yy[gain],minlength=C).tolist(),'saved_correct_harmed':torch.bincount(yy[harm],minlength=C).tolist(),'counts':cnt.tolist()})
 nodamage=all(samecase[-1]['saved_correct_harmed'][c]==0 for c in tiny);method=base['mean_macro']>=.5 and base['final_macro']>=.5 and base['final_tail']>=1 and base['mean_tail']>=-1e-8 and sum(r[s,'M']['modal']['macro']>=r[s,'N0']['modal']['macro']-1e-8 and r[s,'M']['modal']['tail']>=r[s,'N0']['modal']['tail']-1e-8 for s in range(6))>=4 and nodamage
 control=all(min(delta[a].values())>1e-8 for a in ['C','E','Nfull','SEARCH']);credit=min(delta['H'].values())>1e-8
 result={'method_gate':bool(method),'strong_control_gate':bool(control),'risk_credit_gate':bool(credit),'necessary_gate':bool(method and control and credit),'mean_modal':summary,'final':final,'M_minus_controls':delta,'tiny_no_additional_correct_case_damage':nodamage,'rescues_harms':samecase,'TEST_reads':0}
 write(root/'ANALYSIS.json',result);return result
