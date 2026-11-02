import time,torch,hashlib
from core import *
from train_build import loss_build,init_models
from common import *

def measured(fn):
 t=time.time();c=time.process_time();fn();return {'wall_seconds':time.time()-t,'CPU_seconds':time.process_time()-c}
def run(cfg):
 torch.manual_seed(911);C=23;N=1713;p0=unit(torch.randn(C,512,dtype=DT));support=[unit(torch.randn(400,512,dtype=DT)) for c in range(C)];st,p0=state(support,list(range(C)),5,cfg['train_counts'],[0]*13+[1]*2+[2]*2+[3]*2+[4]*2+[5]*2);qz=unit(torch.randn(N,512,dtype=DT));qy=torch.arange(N)%C;g=Builder()
 def buildpass():
  g.zero_grad();loss_build(g,st,p0,qz,qy,C).backward()
 buildpass();b=measured(buildpass)
 sc=torch.randn(1703,C,3,dtype=DT);y=torch.arange(1703)%C;tail=torch.tensor([c<11 for c in range(C)]);protected=torch.ones(C,dtype=torch.bool);ref=torch.full((C,),.5,dtype=DT);rng=torch.Generator().manual_seed(915);actor=Actor();ast=torch.cat([st,torch.zeros(C,512,dtype=DT),torch.zeros(C,1,dtype=DT)],1);lp=actor(ast).detach();prob=lp.exp();a=sampled(prob,rng);mask=wins(sc,y)
 q=conditional(sc,y,a,tail,protected,ref);risk=measured(lambda:conditional(sc,y,a,tail,protected,ref));ce=measured(lambda:conditional(sc,y,a,tail,protected,ref,kind='CE'))
 def four():
  for _ in range(4):
   actor.zero_grad();ll=actor(ast);v,_=exact_B(ll.exp(),mask,y,tail);loss=-v/3+risk_loss(ll,lp,prob,q)/3+.04*kl(ll);loss.backward()
 four();updates=measured(four)
 # Worst stage sizes applied to EVERY stage/episode and safety factor4; include I/O/eval reserve.
 native=__import__('json').loads((__import__('pathlib').Path(cfg['output'])/'NATIVE.json').read_text());projected=4*(b['wall_seconds']*3072*4+384*(risk['wall_seconds']*3+ce['wall_seconds']*4+updates['wall_seconds']*6))+native['four_2row_full_frozen_encode_wall']/8*10188*4+7200
 remain=cfg['deadline']-time.time();receipt={'passed':projected<remain-90,'projected_remaining_seconds':projected,'remaining_seconds':remain,'build_episode':b,'risk_conditional_G8_C23_N1703':risk,'CE_conditional':ce,'four_actor_updates':updates,'safety_factor':4,'extra_IO_search_seal_eval_reserve_seconds':7200,'synthetic_only':True,'new_optimizer_steps':0,'VAL_TEST':0}
 write(__import__('pathlib').Path(cfg['output'])/'BUDGET.json',receipt);return receipt

def initial(cfg):
 models,_=init_models();torch.manual_seed(74002);actor=Actor();record={}
 for name,m in list(models.items())+[('actor',actor)]:
  data=b''.join(t.detach().numpy().tobytes() for t in m.state_dict().values());record[name]={'parameters':sum(p.numel() for p in m.parameters()),'initial_small_parameter_SHA256':hashlib.sha256(data).hexdigest()}
 save(__import__('pathlib').Path(cfg['output'])/'INITIAL_PARAMETERS.private.pt',{'builders':{k:m.state_dict() for k,m in models.items()},'actor':actor.state_dict()});write(__import__('pathlib').Path(cfg['output'])/'INITIAL_PARAMETERS.json',record);return record
