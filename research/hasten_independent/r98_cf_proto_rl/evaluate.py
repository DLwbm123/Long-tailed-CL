"""All fixed endpoints sealed -> S-only F -> all-TRAIN deployment -> VAL once."""
import json,csv,torch
from pathlib import Path
from common import write,save,guard
from prototype_policy import Policy,build_state,build_prototypes,unit
from data_contract import require_seal
MODES=('T','SUP0','SC','RLP','CF','GLOBAL','PATH')

def predictions(c,s,support,y,query):
 n=sum(map(len,c['stages'][:s+1]));birth=[b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl];parts=[support[y==k] for k in range(n)];assert all(len(z) for z in parts);ep=build_state(parts,c['train_counts'][:n],birth,s);protos={'T':ep['p0']}
 for arm in MODES[1:]:
  m=Policy().cuda();step=512 if arm=='SUP0' else 256;v=torch.load(Path(c['output'])/f's{s}_{arm}_{step}.private.pt',map_location='cpu',weights_only=False);m.load_state_dict(v['model'],strict=True)
  with torch.no_grad():protos[arm]=build_prototypes(m(ep['state'])[None],ep)[0]
 return protos

def run(c,meta,counter):
 require_seal(c);r=Path(c['output']);rows,y,roles=meta;F=[];deploy=[]
 with torch.no_grad():
  for s in range(6):
   guard(c);pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False);n=sum(map(len,c['stages'][:s+1]));si=sorted(i for i in roles['S'] if y[i]<n);fi=sorted(i for i in roles['F'] if y[i]<n);z=pack['z'][si].cuda();q=pack['z'][fi].cuda();ps=predictions(c,s,z,y[si].cuda(),q);pred={a:(20*q@p.T).argmax(1).cpu() for a,p in ps.items()};save(r/f'F_predictions_s{s}.private.pt',{'labels':y[fi],'indices':fi,'predictions':pred,'support':'S-full only; T means T_S'});counter['F_rows']+=len(fi)*len(MODES);counter['cache_feature_rows_accessed']+=len(si)+len(fi);F.append({'stage':s,'query_rows':len(fi),'T_is_T_S':True})
  write(r/'F_READOUT_COMPLETE.json',{'stages':F,'after_36_endpoints_seal':True,'no_training_after_F':True})
  for s in range(6):
   guard(c);pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False);ids=pack['ids'];z=pack['z'][ids].cuda();ps=predictions(c,s,z,y[ids].cuda(),None);h=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);err=float((ps['T'].cpu()-h['prototypes']).abs().max());assert err<=2e-5,('ALL_TRAIN_TEACHER_BINDING',s,err);save(r/f'DEPLOY_s{s}.private.pt',{k:v.cpu() for k,v in ps.items()});deploy.append({'stage':s,'teacher_proto_maxabs':err,'support_rows':len(ids)});counter['cache_feature_rows_accessed']+=len(ids)
 write(r/'DEPLOYMENT_SEAL.json',{'stages':deploy,'mean_action_fixed':True,'no_VAL_read_before_this_seal':True})
 from encoder import make_base,load_teacher,visual
 from data import loader
 base,tf,_=make_base(c,'cuda');base.requires_grad_(False);vr=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert len(vr)==1698 and all(x['split']=='val' for x in vr);vy=torch.tensor([c['order'].index(int(x['original_label'])) for x in vr]);counter['VAL_image_encoder_rows']=0
 with torch.no_grad():
  for s in range(6):
   guard(c);h=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);load_teacher(base,h);ids=[i for i,v in enumerate(vy.tolist()) if v<h['seen']];zs=[];at=[]
   for x,j in loader(c,vr,ids,tf):guard(c);zs.append(visual(base,x.cuda()).cpu());at+=j.tolist();counter['image_encoder_rows']+=len(j);counter['VAL_image_encoder_rows']+=len(j)
   assert ids==at;z=torch.cat(zs).cuda();ps=torch.load(r/f'DEPLOY_s{s}.private.pt',map_location='cpu',weights_only=False);pred={a:(20*z@v.cuda().T).argmax(1).cpu() for a,v in ps.items()};counter['VAL_rows']+=len(ids)*len(MODES);save(r/f'VAL_predictions_s{s}.private.pt',{'labels':vy[ids],'indices':ids,'ids':[vr[i]['sample_id'] for i in ids],'components':[vr[i]['identity_component'] for i in ids],'predictions':pred});write(r/'STATUS.json',{'phase':'EVALUATE','stage_complete':s+1,'counts':counter})
 write(r/'EVAL_COMPLETE.json',{'stages':6,'arms':list(MODES),'counts':counter,'sealed_before_F_VAL':True,'TEST':0})
