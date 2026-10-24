import os,json,subprocess
from pathlib import Path
root=Path(os.environ['RUN_ROOT'])
for ds,mode in [('NIH','native'),('NIH','flat'),('NIH','factorized'),('ISIC','factorized')]:
 d=root/'fixtures'/ds/mode;d.mkdir(parents=True,exist_ok=True);(d/'runtime').mkdir(exist_ok=True);c=json.loads((root/'audit'/mode/ds/'0/config.private.json').read_text());c['output']=str(d);(d/'config.private.json').write_text(json.dumps(c));env=os.environ.copy();env.update(RUN_CONFIG=str(d/'config.private.json'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 code="""import torch
import torch.nn.functional as F
from pathlib import Path
from worker import ns,cfg
c=cfg;bank=torch.load(Path(c['cache_root'] if c['dataset']=='NIH' else c['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
x=bank['x'];text=bank['text'];x=F.normalize(x,dim=1) if c['dataset']=='NIH' else x;text=F.normalize(text,dim=1) if c['dataset']=='NIH' else text
ns['seed_all'](74002);ns['install_feature_model'](ns['INC'],text,bank['prompts'],bank['logit_scale']);args=ns['make_args'](74002,'H-unit');l=ns['CheckedLearner'](args);l.seed=74002;l.arm='H-unit';l._cur_task=0;l._total_classes=len(ns['STAGES'][0]);l._known_classes=0;n=l._total_classes;l.start_audit();net=l._network;net.update_fc(n);net.update_task();net.to('cuda')
# A synthetic class-balanced small fixture uses existing TRAIN vectors only, no optimization.
xx=x[:2*n];yy=torch.arange(n).repeat_interleave(2);l.data_manager=ns['DataManager'](xx,yy,list(range(2*n)),[])
with torch.no_grad():net.fc.weight.copy_(xx.cuda().reshape(n,2,512).mean(1))
l._prepare_decision();before={k:v.clone() for k,v in net.state_dict().items()};raw=xx.cuda();leaf=net.model.encode_text(net.tokenizer([ns['LEAF_TEMPLATE'].format(v) for v in ns['NAMES'][:n]]).cuda());hi=net.trans_hyper(net.Image_encode(raw));ht=net.trans_hyper(net.Text_encode(leaf));z=-ns['GEO'].pairwise_dist(hi,ht,net.get_curv())
if c['decision_mode']!='native':loss=l._decision_loss(raw,None,z,yy.cuda());loss.backward();l._before_step();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
if c['decision_mode']=='factorized':
 p,_,_=ns['_decision_posterior'](net,raw,z*net.logit_scale.exp());assert torch.allclose(p.exp().sum(1),torch.ones(len(p),device='cuda',dtype=p.dtype),atol=1e-6)
ns['seed_all'](74002);other=ns['CheckedLearner'](args);other._network.update_fc(n);other._network.update_task();other._network.to('cuda');other._network.load_state_dict(before,strict=True);assert all(torch.equal(v,other._network.state_dict()[k]) for k,v in before.items())
if c['decision_mode']=='factorized':
 keep=net.coarse_prototypes.detach().clone();net.update_fc(n+2);net.update_task();net.to('cuda');assert torch.equal(net.coarse_prototypes,keep)
print('FINITE_NOOPT_STATE_RNG_STRICTLOAD_AND_GROWTH_PASSED')
"""
 (Path('/tmp/q1080')/'fixture.py').write_text(code)
 with (d/'preflight.private.log').open('w') as f:p=subprocess.run(['/tmp/q108v/bin/python','/tmp/q1080/fixture.py'],env=env,cwd=d/'runtime',stdout=f,stderr=subprocess.STDOUT,timeout=90)
 assert p.returncode==0,(ds,mode)
(root/'PREFLIGHT_ACCEPTANCE.json').write_text(json.dumps(dict(finite_backward=True,probability_normalized=True,strict_state_load=True,coarse_growth_preserves_weights=True,epoch_observation_state_rng_restored=True,optimizer_steps=0,fixtures_only=True,not_scientific_success=True),indent=2));print('PREFLIGHT_PASSED')
