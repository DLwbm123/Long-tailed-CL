"""Frozen first-arrival48-root Qfit-to-Qaudit gate; no actor training or VAL."""
import json,time
from pathlib import Path
from shadow import episode,root,feedback,gen
from policy import prior
import torch
from common import write,guard,save

def run(c,data,roles,rows,counts,counter):
 out=[];r=Path(c['output']);folder=r/'pilot';folder.mkdir()
 for j in range(24):
  guard(c);seed=74002*100000+j*101+8100;e,skip=episode(data,roles,rows,counts,seed,j)
  if skip:out.append(dict(root=j,**skip));continue
  ctx=root(e,counter,'A',c);scores,raw=feedback(e,ctx,counter,'A',c);fit=scores['FIT'];a=scores['AUDIT'];chosen=max(range(len(fit['R'])),key=lambda k:fit['R'][k]);p0=prior(len(ctx['mems']),ctx['x'].device);draws=torch.multinomial(p0,32,replacement=True,generator=gen(seed+80000,ctx['x'].device));record={'prior_draw32_counts':torch.bincount(draws,minlength=len(p0)).cpu().tolist(),'candidate_keys':ctx['candidate_keys'],'anchor_keys':ctx['anchor_keys'],'root':j,'chosen_fit':chosen,'candidate_count':len(ctx['mems']),'nontrivial':len(ctx['mems'])>1,'old_scarce_evaluable':a['old_scarce_evaluable'],'audit_delta_R':a['R'][chosen]-a['R'][0],'audit_delta_U':a['U'][chosen]-a['U'][0],'audit_H_selected':a['H'][chosen],'fit_tie_count':sum(v==max(fit['R']) for v in fit['R']),'audit_oracle_delta_R':max(a['R'])-a['R'][0],'coverage':e['coverage'],'actual_B':ctx['actual_B'],'C0':ctx['C0'],'maximum_age':e['prefix_phases']-1,'proposal_audit':ctx['proposal_audit'],'scores':scores,'raw_query_counts':raw};out.append(record);write(folder/'ROOTS.private.json',out);write(r/'ACTIVE_COUNTS.private.json',{'phase':'A_FEEDBACK_TRANSFER','root':j,'counts':counter});del e,ctx,scores,raw
 valid=[x for x in out if 'candidate_count' in x];non=[x for x in valid if x['nontrivial']];old=[x for x in valid if x['old_scarce_evaluable']];mean=lambda key:sum(x[key] for x in valid)/len(valid) if valid else None
 gate={'passed':len(non)>=16 and len(old)>=8 and mean('audit_delta_R') is not None and mean('audit_delta_R')>=.005 and mean('audit_delta_U')>=0 and mean('audit_H_selected')<=.005,'roots_requested':24,'evaluable_roots':len(valid),'nontrivial_evaluable_roots':len(non),'old_scarce_evaluable_roots':len(old),'mean_audit_delta_R':mean('audit_delta_R'),'mean_audit_delta_U':mean('audit_delta_U'),'mean_audit_H_selected':mean('audit_H_selected'),'thresholds':{'minimum_nontrivial':16,'minimum_old_scarce':8,'mean_delta_R_min':.005,'mean_delta_U_min':0,'mean_H_max':.005},'decision_from_Qfit_only':True,'actor_optimizer':0,'VAL_TEST':0,'empty_and_noop_roots_included_in_gain_means':True,'not_independent_patients':True}
 write(r/'PILOT_GATE.json',gate);write(r/'A_COMPLETE.json',{'gate':gate,'counts':counter});return gate
