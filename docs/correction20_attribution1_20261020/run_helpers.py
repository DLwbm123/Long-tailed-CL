import json
import numpy as np
from core import weights

def write(p,obj):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)


def contrasts(scores,y,g):
 p={k:v.argmax(1)==y for k,v in scores.items()};w=weights(y,g);w/=w.sum();iw=1/(20*np.bincount(y,minlength=20)[y]);out={}
 for arm in scores:
  for ref in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction','cls_plain_correction']:
   if arm==ref:continue
   rescue=p[arm]&~p[ref];harm=~p[arm]&p[ref]
   out[arm+'-'+ref]=dict(rescue=int(rescue.sum()),harm=int(harm.sum()),delta_BA=float(iw@(rescue.astype(float)-harm)),delta_PP=float(w@(rescue.astype(float)-harm)),PP_rescue=float(w@rescue),PP_harm=float(w@harm))
 c=p['cls768_ridge'];j=p['scale_joint'];masks={'both_correct':c&j,'CLS_only':c&~j,'joint_only':~c&j,'both_wrong':~c&~j}
 return out,{name:dict(images=int(m.sum()),correct={arm:int((ok&m).sum()) for arm,ok in p.items()}) for name,m in masks.items()}

