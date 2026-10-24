"""One synthetic CPU check of new word objective, physical pixels, and cohorts."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from core import Model,shuffle_receipt,corrupt,crop,patchify
import numpy_word
from word_data import discover,word_draw

def main(path):
    torch.set_num_threads(2);torch.manual_seed(74007);tick=time.time();gates=[]
    def check(name,values,passed):
        gates.append(dict(check=name,values=values,PASS=bool(passed)));Path(path).write_text(json.dumps(dict(state='RUNNING',gates=gates),indent=2)+'\n');assert passed,name
    raw=torch.rand(1,3,224,224);receipt=shuffle_receipt(74007);y=corrupt(raw,receipt);expected=numpy_word.corrupt(raw.numpy(),receipt);pixel=float(abs(y.numpy()-expected).max());check('physical_RGB_shared_shuffle',dict(error=pixel,permutations=len(receipt['permutations'])),pixel==0 and all(sorted(p)==list(range(256)) for p in receipt['permutations']))
    error=0.
    for top,left in [(4,4),(108,108),(55,71)]:error=max(error,float(abs(crop(raw,top,left).numpy()-numpy_word.crop(raw.numpy(),top,left)).max()))
    check('independent_bilinear_crop_FP32',dict(max_error=error,tolerance=2e-7),error<=2e-7)
    model=Model(torch.nn.Identity(),torch.randn(10,1536,dtype=torch.float64));z=torch.randn(3,197,768,dtype=torch.float64,requires_grad=True);target=torch.rand(3,3,224,224,dtype=torch.float64);labels=torch.tensor([0,17,99]);loss=model.objective_from_tokens(z,target,labels,'VISUAL_WORD_CLASS_RESTORATION');loss.backward()
    nz=z.detach().numpy();W=model.decoder.weight.detach().numpy();b=model.decoder.bias.detach().numpy();Q=model.word_head.weight.detach().numpy();a=model.word_head.bias.detach().numpy();patches=numpy_word.patchify(target.numpy());pred=nz[:,1:]@W.T+b;dp=2*(pred-patches)/pred.size;gw=dp.reshape(-1,768).T@nz[:,1:].reshape(-1,768);gb=dp.sum((0,1));gz=np.zeros_like(nz);gz[:,1:]=dp@W
    s=np.concatenate((nz[:,0],nz[:,1:].mean(1)),1);r=np.linalg.norm(s,axis=1,keepdims=True);u=s/r;logits=u@Q.T+a;logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True);dl=prob.copy();dl[np.arange(3),labels.numpy()]-=1;dl*=.01/3;gQ=dl.T@u;ga=dl.sum(0);gu=dl@Q;gs=(gu-u*(gu*u).sum(1,keepdims=True))/r;gz[:,0]+=gs[:,:768];gz[:,1:]+=gs[:,None,768:]/196
    grad=max(float(abs(gz-z.grad.numpy()).max()),float(abs(gw-model.decoder.weight.grad.numpy()).max()),float(abs(gb-model.decoder.bias.grad.numpy()).max()),float(abs(gQ-model.word_head.weight.grad.numpy()).max()),float(abs(ga-model.word_head.bias.grad.numpy()).max()));reference=numpy_word.objective(nz,target.numpy(),labels.numpy(),'VISUAL_WORD_CLASS_RESTORATION',dict(weight=W,bias=b),dict(weight=Q,bias=a),model.disease_head.detach().numpy());objective_error=abs(float(loss)-reference);check('new_word_plus_restoration_objective_and_analytic_gradients',dict(objective_error=objective_error,gradient_error=grad),objective_error<=1e-12 and grad<=1e-12)
    means={str(i):v/v.norm() for i,v in enumerate(torch.randn(129,1536,dtype=torch.float64))};means['1']=means['0'].clone();words=discover(means,74007,lambda v:gates.append(v));groups={g:[2*i,2*i+1] for i,g in enumerate(sorted(means))};draw=word_draw(words,groups,74008);check('real_shape_synthetic_discovery_and_draw',dict(words=len(words),patients_per_word=64,draws=len(draw)),len(words)==100 and all(len(w['patients'])==len(set(w['patients']))==64 and w['anchor'] in w['patients'] for w in words) and len(draw)==16 and all(4<=v['top']<=108 and 4<=v['left']<=108 for v in draw))
    out=dict(state='PASS',gates=gates,CPU_objective_forward=1,CPU_objective_backward=1,optimizer_updates=0,NIH_images=0,GPU_calls=0,real_native_models=0,wall_seconds=time.time()-tick);Path(path).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='gates'}))
if __name__=='__main__':main(sys.argv[1])
