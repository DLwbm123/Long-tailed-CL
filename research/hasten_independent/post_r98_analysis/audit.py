"""Bounded S/B-only fixed-endpoint diagnostic, never optimizer or image encoder."""
import os,sys,json,time,resource,signal,math
from pathlib import Path
start=time.time();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(4000,4000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));signal.alarm(1800);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]))
source=Path(os.environ['FROZEN_SOURCE']);out=Path(os.environ['AUDIT_OUTPUT']);sys.path.insert(0,str(source))
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
from prototype_policy import Policy,build_state,build_prototypes,gaussian_KL,SIGMA
from utility import utility,class_values
from train import flat_grad
cfg=json.loads((source/'RUN_CONFIG.private.json').read_text());src=json.loads((source/'SOURCE_BINDING.private.json').read_text());records=[];counts={'policy_state_forwards':0,'raw_B_score_rows':0,'utility_calls':0,'PATH_aux_backward':0,'other_backward':0,'optimizer':0,'image_encoder':0,'new_F_VAL_TEST_readout':0}

def vector_stats(a,b):return {'a_norm':float(a.norm()),'b_norm':float(b.norm()),'cosine':float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))}
def metric(z,y,raw):
 n=z.shape[-1];cor=torch.bincount(y[z.argmax(-1)==y],minlength=n);den=torch.bincount(y,minlength=n);rec=cor.double()/den;tail=torch.tensor(raw)<250;return {'micro':float(cor.sum().double()/len(y)*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100)}
for s in range(6):
 assert time.time()-start<1700
 n=sum(map(len,cfg['stages'][:s+1]));pack=torch.load(Path(cfg['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False);y=pack['labels'];si=sorted(i for i in src['roles']['S'] if y[i]<n);bi=sorted(i for i in src['roles']['W'] if y[i]<n);assert not set(si)&set(bi);sy=y[si];by=y[bi];support=[pack['z'][si][sy==c] for c in range(n)];birth=[j for j,cl in enumerate(cfg['stages'][:s+1]) for _ in cl];ep=build_state(support,cfg['train_counts'][:n],birth,s);query=pack['z'][bi];nB=torch.bincount(by,minlength=n).float();protected=torch.tensor([b<s or cfg['train_counts'][c]<250 for c,b in enumerate(birth)]);tail=torch.tensor(cfg['train_counts'][:n])<250
 models={};det={};protos={};stats={}
 with torch.no_grad():
  for arm in ['SUP0','SC','RLP','CF','GLOBAL','PATH']:
   m=Policy();v=torch.load(source/f's{s}_{arm}_{512 if arm=="SUP0" else 256}.private.pt',map_location='cpu',weights_only=False);m.load_state_dict(v['model'],strict=True);models[arm]=m;mu=m(ep['state']);p=build_prototypes(mu[None],ep)[0];z=20*query@p.T;protos[arm]=p;det[arm]=z;counts['policy_state_forwards']+=1;counts['raw_B_score_rows']+=len(by);stats[arm]={'B_deterministic_metrics':metric(z,by,cfg['train_counts'][:n]),'mu_abs_max':float(mu.abs().max())}
  z=20*query@ep['p0'].T;stats['T_S']={'B_deterministic_metrics':metric(z,by,cfg['train_counts'][:n])};counts['raw_B_score_rows']+=len(by);ref=class_values(det['SUP0'][None],by)[0]
  for arm in stats:
   zz=z if arm=='T_S' else det[arm];stats[arm]['B_deterministic_U']=float(utility(zz[None],by,ref,protected)[0]);counts['utility_calls']+=1
  for arm in ['SUP0','CF','GLOBAL','PATH']:
   m=models[arm];mu=m(ep['state']);counts['policy_state_forwards']+=1;us=[];mac=[]
   for j in range(32):
    gen=torch.Generator().manual_seed(989000000+100003*s+j);eps=torch.randn(8,n,4,generator=gen);pp=build_prototypes(mu[None]+SIGMA*eps,ep);zz=20*torch.einsum('qd,gcd->gqc',query,pp);us+=utility(zz,by,ref,protected).tolist();counts['raw_B_score_rows']+=8*len(by);counts['utility_calls']+=8
   v=torch.tensor(us,dtype=torch.double);stats[arm]['B_sampled_U_mean']=float(v.mean());stats[arm]['B_sampled_U_std']=float(v.std(correction=1));stats[arm]['B_sampled_U_MC_standard_error']=float(v.std(correction=1)/math.sqrt(len(v)));stats[arm]['sample_minus_deterministic_U']=float(v.mean())-stats[arm]['B_deterministic_U']
 # Differentiate the same stochastic target at the final CF point, not a new trained arm.
 m=models['CF'];refm=models['SUP0'];mu=m(ep['state']);p=build_prototypes(mu[None],ep)[0];zz=20*query@p.T;counts['policy_state_forwards']+=1;counts['raw_B_score_rows']+=len(by);ce=F.cross_entropy(zz+nB.log()[None],by);gce=flat_grad(ce,m);counts['other_backward']+=1
 mu=m(ep['state']);rm=refm(ep['state']).detach();kl=.04*gaussian_KL(mu,rm,ep['active']);gkl=flat_grad(kl,m);counts['policy_state_forwards']+=2;counts['other_backward']+=1;gr=[]
 for j in range(32):
  mu=m(ep['state']);gen=torch.Generator().manual_seed(989000000+100003*s+j);eps=torch.randn(8,n,4,generator=gen);pp=build_prototypes(mu[None]+SIGMA*eps,ep);zz=20*torch.einsum('qd,gcd->gqc',query,pp);loss=-utility(zz,by,ref,protected).mean();gr.append(flat_grad(loss,m).double());counts['policy_state_forwards']+=1;counts['raw_B_score_rows']+=8*len(by);counts['utility_calls']+=8;counts['PATH_aux_backward']+=1
 g=torch.stack(gr);mean=g.mean(0);grad={'MC_PATH_expected_reward_loss_gradient_vs_CE':vector_stats(mean,gce.double()),'weighted_Gaussian_KL_vs_CE':vector_stats(gkl,gce),'reward_vs_weighted_KL':vector_stats(mean,gkl.double()),'mean_reward_gradient_MC_norm_standard_error':float(torch.sqrt(g.var(0,correction=1).sum()/32)),'weighted_KL_value':float(kl),'CE_value':float(ce),'unclipped_raw_parameter_gradients':True}
 base=det['SUP0'];dz=det['CF']-base;lead=base.topk(2,dim=1).values;gap=lead[:,0]-lead[:,1];maxchange=dz.abs().max(1).values;flips=(base.argmax(1)!=det['CF'].argmax(1));bounds=gap>2*maxchange
 dep=torch.load(source/f'DEPLOY_s{s}.private.pt',map_location='cpu',weights_only=False);dS=protos['CF']-protos['SUP0'];dAll=dep['CF']-dep['SUP0'];sn=dS.norm(dim=1);an=dAll.norm(dim=1);keep=(sn>1e-10)&(an>1e-10);cos=(dS*dAll).sum(1)/(sn*an).clamp_min(1e-30)
 change={'B_CF_vs_SUP0_prediction_changes':int(flips.sum()),'B_rows':len(by),'B_median_abs_logit_change':float(dz.abs().median()),'B_max_abs_logit_change':float(dz.abs().max()),'B_certified_unchanged_by_gap_fraction':float(bounds.float().mean()),'B_SUP0_median_top2_gap':float(gap.median()),'S_class_proto_CF_SUP0_mean_L2':float(sn.mean()),'S_class_proto_CF_SUP0_max_L2':float(sn.max()),'deployed_class_proto_CF_SUP0_mean_L2':float(an.mean()),'deployed_class_proto_CF_SUP0_max_L2':float(an.max()),'universal_unit_query_deployment_logit_change_upper':float(20*an.max()),'S_to_deployment_correction_direction_cosine_mean':float(cos[keep].mean()),'S_to_deployment_correction_direction_negative_classes':int((cos[keep]<0).sum()),'nonzero_correction_classes':int(keep.sum())}
 records.append({'stage':s,'models':stats,'final_CF_gradient_separation':grad,'action_to_boundary':change});print('stage',s,'completed',flush=True)
result={'scope':'Fixed saved SUP0/SC/RLP/CF/GLOBAL/PATH endpoints, six stages, only original S-full/B frozen features. Deterministic and fixed32xG8 diagnostic, no tuning, optimizer, new encoder or F/VAL scoring. Full deployment prototype tensors already saved are compared geometrically only. Not an independent efficacy experiment.','records':records,'counts':counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};out.write_text(json.dumps(result,indent=2,allow_nan=False));print('DONE',result['CPU_seconds'],result['wall_seconds'],flush=True)
