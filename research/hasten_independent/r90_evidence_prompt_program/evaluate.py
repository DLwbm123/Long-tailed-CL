import csv,json
from pathlib import Path
import numpy as np
import torch
import counts
from program import PROGRAMS
from experiment import encode_ids,safe
from common import write

def metrics(correct,total,fullcounts,old):
 recall=correct/total;tail=np.asarray(fullcounts)<250;n=len(total);groups={}
 for name,mask in [('old_head',(np.arange(n)<old)&~tail),('old_tail',(np.arange(n)<old)&tail),('new_head',(np.arange(n)>=old)&~tail),('new_tail',(np.arange(n)>=old)&tail)]:groups[name]=None if not mask.any() else float(recall[mask].mean()*100)
 return {'counts':total.tolist(),'correct':correct.tolist(),'recall':(recall*100).tolist(),'micro':float(correct.sum()/total.sum()*100),'macro':float(recall.mean()*100),'tail':float(recall[tail].mean()*100),'groups':groups}
def run(cfg,base,tf,dictionary):
 r=Path(cfg['output']);seal=json.loads((r/'ALL_STATES_SEALED.json').read_text());assert seal['policy_snapshots']==30 and seal['VAL_reads']==0
 with open(Path(cfg['manifest'])/'val.csv') as f:rows=list(csv.DictReader(f))
 assert all(v['split']=='val' for v in rows);labels=np.array([cfg['order'].index(int(x['original_label'])) for x in rows]);assert len({x['sample_id'] for x in rows})==len(rows)==1698
 features=np.empty((16,len(rows),512),np.float32)
 for k,program in enumerate(PROGRAMS):features[k]=encode_ids(cfg,base,dictionary,rows,list(range(len(rows))),tf,program)
 np.save(r/'VAL_features.private.npy',features,allow_pickle=False);counts.C['VAL_rows']+=16*len(rows);records=[]
 for stage in range(6):
  safe(cfg);n=sum(map(len,cfg['stages'][:stage+1]));old=sum(map(len,cfg['stages'][:stage]));ids=np.flatnonzero(labels<n);yy=labels[ids];total=np.bincount(yy,minlength=n);assert (total>0).all();z=torch.load(r/f'deployment_s{stage}.private.pt',map_location='cpu',weights_only=False);scores=features[:,ids]@z['proto'].transpose(0,2,1);counts.C['classifier_score_rows']+=16*len(ids);correct=np.stack([np.bincount(yy[scores[k].argmax(1)==yy],minlength=n) for k in range(16)]);arms={}
  for name,prob in z['distributions'].items():
   prob=np.asarray(prob,dtype=np.float64);assert np.isfinite(prob).all() and (prob>=0).all() and abs(prob.sum()-1)<1e-12;mode=int(prob.argmax());m=metrics(prob@correct,total,cfg['train_counts'][:n],old);m.update({'modal':metrics(correct[mode],total,cfg['train_counts'][:n],old),'modal_program':list(PROGRAMS[mode]),'probabilities':prob.tolist()});arms[name]=m
  memscores=features[0,ids]@z['all_memory_proto'].T;counts.C['classifier_score_rows']+=len(ids);memcorrect=np.bincount(yy[memscores.argmax(1)==yy],minlength=n);arms['NCM_MEMORY']=metrics(memcorrect,total,cfg['train_counts'][:n],old);arms['NCM_MEMORY']['modal']=dict(arms['NCM_MEMORY'])
  torch.save({'stage':stage,'ids':[rows[i]['sample_id'] for i in ids],'labels':yy,'program_scores':scores,'program_correct':correct,'counts':total,'memory_scores':memscores,'memory_correct':memcorrect},r/f'VAL_scores_s{stage}.private.pt');records.append({'stage':stage,'classes':n,'arms':arms});write(r/'STAGE_METRICS.json',records);write(r/'STATUS.json',{'phase':'EVALUATE','stage_complete':stage+1,'counts':dict(counts.C)})
 write(r/'EVAL_COMPLETE.json',{'stages':6,'all_states_sealed_before_VAL':True,'counts':dict(counts.C)})
