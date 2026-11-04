"""All 24 endpoints sealed, fixed S deployment, F then VAL once, TEST never."""
import csv,json
from pathlib import Path
import torch
from common import write,save,guard
from model import Joint,ARMS,unit,logits,prototypes
from data_contract import load
MODES=('T','A0',*ARMS)

@torch.no_grad()
def deployment(c,s,meta):
    d=load(c,s,meta);out={};r=Path(c['output'])
    for arm in ('A0',*ARMS):
        torch.manual_seed(74002);m=Joint('UNIFORM' if arm=='A0' else arm).cuda()
        if arm!='A0':
            state=torch.load(r/f's{s}_{arm}.private.pt',map_location='cpu',weights_only=False);m.load_state_dict(state['model'],strict=True)
        h=m.adapt(d['S']);parts,b,_=logits(m,h,d);p,_=prototypes(parts,b,d);out[arm]=(m,p)
    save(r/f'DEPLOY_s{s}.private.pt',{'base':d['base'].cpu(),'prototypes':{a:p.cpu() for a,(m,p) in out.items()},'support_ids':d['Sids'],'support_rule':'Fixed original S; no full-TRAIN reconstruction'})
    return d,out

@torch.no_grad()
def predict(z,d,models):
    result={'T':(20*z@d['base'].T).argmax(1).cpu()}
    for a,(m,p) in models.items():result[a]=(20*m.adapt(z)@p.T).argmax(1).cpu()
    return result

def run(c,meta,counts):
    r=Path(c['output']);seal=json.loads((r/'SEAL.json').read_text());assert seal['final_endpoints']==24 and seal['total_optimizer_steps']==21504
    rows,y,roles=meta;deployed=[]
    with torch.no_grad():
        for s in range(6):
            guard(c);d,models=deployment(c,s,meta);C=d['C']
            pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False)
            fi=sorted(i for i in roles['F'] if int(y[i])<C);z=pack['z'][fi].cuda();pred=predict(z,d,models)
            save(r/f'F_predictions_s{s}.private.pt',{'labels':y[fi],'indices':fi,'predictions':pred})
            counts['F_rows']+=len(fi)*len(MODES);deployed.append({'stage':s,'support_rows':len(d['S']),'F_queries':len(fi)})
            del models,d,pack,z
    write(r/'DEPLOYMENT_SEAL.json',{'all_fixed':True,'stages':deployed,'deterministic_weight_logits':True,'fixed_support_no_role_or_scale_switch':True,'VAL_before_this_seal':0})
    from encoder import make_base,load_teacher,visual
    from data import loader
    base,tf,_=make_base(c,'cuda');vr=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert len(vr)==1698 and all(x['split']=='val' for x in vr)
    vy=torch.tensor([c['order'].index(int(x['original_label'])) for x in vr])
    with torch.no_grad():
        for s in range(6):
            guard(c);head=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);load_teacher(base,head)
            d,models=deployment(c,s,meta);ids=[i for i,v in enumerate(vy.tolist()) if v<d['C']];zs=[];at=[]
            for x,j in loader(c,vr,ids,tf):
                guard(c);zs.append(visual(base,x.cuda()).cpu());at+=j.tolist();counts['image_encoder']+=len(j)
            assert at==ids;z=torch.cat(zs).cuda();pred=predict(z,d,models)
            save(r/f'VAL_predictions_s{s}.private.pt',{'labels':vy[ids],'indices':ids,'components':[vr[i]['identity_component'] for i in ids],'predictions':pred})
            counts['VAL_rows']+=len(ids)*len(MODES);write(r/'STATUS.json',{'phase':'EVALUATE','stage_complete':s+1,'counts':counts})
            del models,d,z
    write(r/'EVAL_COMPLETE.json',{'stages':6,'modes':list(MODES),'counts':counts,'TEST':0,'all_states_sealed_before_F_VAL':True})
