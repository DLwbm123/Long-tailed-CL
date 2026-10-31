import time,resource,torch,json
from pathlib import Path
from model import Adapter,unit

def run(cfg):
 start=time.time();cpu=time.process_time();torch.set_num_threads(4);torch.manual_seed(74002);receipt={};features=unit(torch.randn(9,512));y=torch.tensor([0,1,1,1,2,2,2,2,2]);m=Adapter('FULL');m.add();p,w,rl=m.prototypes(features,y,3)
 for c in range(3):assert torch.allclose(w[y==c].sum(),torch.tensor(1.))
 assert w[0]==1;assert torch.allclose(p[0],features[0]);assert torch.allclose(w[y==2],torch.full((5,),.2))
 q=unit(torch.randn(2,512));text=unit(torch.randn(3,512));s,d,b=m.logits([q],text,p,[0,0,0]);assert s.shape==(2,3) and torch.isfinite(s).all();s2=m.logits([q],text,p,[0,0,0])[0];assert torch.equal(s,s2)
 opt=torch.optim.AdamW([z for z in m.parameters() if z.requires_grad],lr=1e-4);loss=torch.nn.functional.cross_entropy(s,torch.tensor([0,2]))+.1*rl+.01*b;opt.zero_grad();loss.backward();assert any(z.grad is not None and z.grad.abs().sum()>0 for z in m.policy.parameters());assert all(z.grad is None for z in m.reference.parameters());opt.step();m.update_reference();m.add();assert all(not z.requires_grad for z in m.branches[0].parameters());assert all(z.requires_grad for z in m.branches[1].parameters())
 # Exact one-exemplar class weights have no policy-controlled motion.
 pp,ww,_=m.prototypes(features[:3],torch.arange(3),3);assert torch.equal(ww,torch.ones(3));assert torch.allclose(pp,features[:3],atol=2e-7)
 rec={'passed':True,'toy_optimizer_steps':1,'new_real_encoder_rows':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};Path(cfg['output'],'SOFTWARE_CHECK.json').write_text(json.dumps(rec,indent=2));return rec
