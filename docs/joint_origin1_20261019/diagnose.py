"""Descriptive decompositions of already-fitted scores; no fitting."""
import torch


def ba(labels,pred,classes):
 return sum(float((pred[labels==c]==c).double().mean()) for c in classes)/len(classes)


def stage_diagnostics(saved,previous):
 labels=saved['labels'];seen=saved['seen'];scores=saved['scores'];pred={a:torch.tensor(seen)[s.argmax(1)] for a,s in scores.items()}
 cls=pred['cls768_ridge']==labels;patch=pred['patchmean768_ridge']==labels
 masks={'both_correct':cls&patch,'CLS_only':cls&~patch,'patch_only':~cls&patch,'both_wrong':~cls&~patch}
 groups={name:dict(count=int(mask.sum()),correct={a:int(((p==labels)&mask).sum()) for a,p in pred.items()}) for name,mask in masks.items()}
 competition={}
 if previous is not None:
  old=previous['seen'];mask=torch.tensor([int(y) in old for y in labels]);y=labels[mask]
  assert torch.equal(y,previous['labels']) and seen[:len(old)]==old
  for arm,s in scores.items():
   pold=torch.tensor(old)[s[mask,:len(old)].argmax(1)];pall=pred[arm][mask]
   pp=torch.tensor(old)[previous['scores'][arm].argmax(1)]
   before=ba(y,pp,old);internal=ba(y,pold,old);allba=ba(y,pall,old)
   stolen=(pold==y)&(pall!=y)
   assert all(int(p) not in old for p in pall[stolen])
   loss=internal-allba;change=internal-before;total=allba-before
   assert loss>=-1e-12 and abs(total-(change-loss))<1e-12
   competition[arm]=dict(previous_old_BA=before,current_restricted_old_BA=internal,current_all_old_BA=allba,internal_change=change,competition_loss=loss,total_change=total,new_class_stolen_correct=int(stolen.sum()))
 return dict(groups=groups,competition=competition)


def trajectories(records,order):
 arms=list(records[-1]['metrics']);out=[]
 for c in order:
  born=next(i for i,r in enumerate(records) if c in r['seen']);data={}
  for a in arms:
   vals=[r['metrics'][a]['recall'][r['seen'].index(c)] for r in records[born:]]
   ref=[r['metrics']['cls768_ridge']['recall'][r['seen'].index(c)] for r in records[born:]]
   advantage=[x-y for x,y in zip(vals,ref)]
   data[a]=dict(recalls=vals,advantage_over_CLS=advantage,birth_advantage=advantage[0],final_advantage=advantage[-1],advantage_change=advantage[-1]-advantage[0])
  out.append(dict(class_id=c,birth_stage=born,arms=data))
 return out
