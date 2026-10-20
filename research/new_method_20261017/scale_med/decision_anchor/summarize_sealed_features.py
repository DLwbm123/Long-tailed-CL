import os,signal,resource,time,json,pathlib
signal.alarm(60)
resource.setrlimit(resource.RLIMIT_AS,(12*2**30,12*2**30))
resource.setrlimit(resource.RLIMIT_FSIZE,(512*2**20,512*2**20))
start=time.monotonic()
import torch
torch.set_num_threads(4)
r=pathlib.Path(os.environ['ROUND_ROOT']);assert time.time()<float(os.environ['ROUND_DEADLINE'])
cfg=json.loads((r/'fit.config.private.json').read_text())
s=torch.load(r/'train/protection.private.pt',map_location='cpu',weights_only=False)
o=torch.load(pathlib.Path(cfg['reference'])/'inputs/0.private.pt',map_location='cpu',weights_only=False);tr=~o['valid'];y=o['y'][tr]
assert int(tr.sum())==2555 and json.loads((r/'fit/MODEL_SEAL.json').read_text())['total_fits']==15
def transform(g):
 t=s['mapper'];return (((g-t['gn'][0])/t['gn'][1])[:,t['ids']]-t['on'][0])/t['on'][1]
z0=transform(o['g'][tr].double());rows={}
for a in ['E','L','R','D']:
 seal=json.loads((r/('fit/'+a+'.FEATURE_SEAL.json')).read_text());assert seal['rows']==3236 and seal['one_pass']
 g=torch.load(r/('fit/'+a+'.g.private.pt'),map_location='cpu',weights_only=False);delta=transform(g[tr].double())-z0
 energies={'total_drift':delta.square().sum(1)}
 for k in ['D','R','S']:
  denom=s['t'] if k=='S' else s['t']*s['r']/s['m'];energies['L' if k=='S' else k]=(delta@s[k]).square().sum(1)/denom
 assert all(torch.isfinite(v).all() and (v>=0).all() for v in energies.values())
 rows[a]={k:{'uniform_all_training_mean':float(v.mean()),'class_means':[float(v[y==c].mean()) for c in range(4)]} for k,v in energies.items()}
u=resource.getrusage(resource.RUSAGE_SELF)
d={'phase':'POST_COMPLETION_CPU_SEALED_FEATURE_AGGREGATION','rows':2555,'classes':[int((y==c).sum()) for c in range(4)],'summary':rows,'unweighted_means_not_training_group_weighted':True,'encoder_forwards':0,'parameter_updates':0,'test_used':False,'future_used':False,'full_dataset_CE_available':False,'CE_source':'first/last100 shared scheduled training batches in train_partial.json','CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.monotonic()-start,'peak_RSS_bytes':u.ru_maxrss*1024,'source_commit':'d13da4d','posthoc_descriptive_only':True}
assert time.time()<float(os.environ['ROUND_DEADLINE'])
(r/'results/TRAINING_GEOMETRY.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps(d))
