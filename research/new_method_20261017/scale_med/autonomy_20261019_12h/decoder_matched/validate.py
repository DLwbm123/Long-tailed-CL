import os,json,sys,math,time
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['root']);o=Path(cfg['output']);sys.path.insert(0,str(r));from space import used
a=json.loads((o/'aggregate.json').read_text());p=torch.load(o/'predictions.private.pt',weights_only=False);prior=torch.load(cfg['inherited_predictions'],weights_only=False)
assert len(a['fits'])==11 and all(c['status']=='CERTIFIED' for c in a['fits']);assert len(a['artifact_bytes'])==9 and max(a['artifact_bytes'].values())<=19058800
assert len(p['scores'])==162 and len(prior['scores'])==153;assert all(torch.equal(p['scores'][k],v) for k,v in prior['scores'].items());assert all(torch.isfinite(v).all() for v in p['scores'].values())
assert torch.equal(p['y'],prior['y']) and p['groups']==prior['groups'];assert len(p['y'])==586 and len(set(p['groups']))==500
assert len(a['comparisons']['C_Matched']['vs_ridge'])==4 and a['feature_rows']==6112 and json.loads((o/'SELECTION_SEAL.json').read_text())['alpha']==.01
for st in range(3):
 assert torch.equal(p['scores']['G_Matched',st].argmax(1),p['scores']['G_ridge',st].argmax(1))
 for name in ['G_Matched','P_Matched','C_Matched']:
  pred=p['scores'][name,st].argmax(1);yy=p['y'];correct=[int(((yy==c)&(pred==c)).sum()) for c in range(st+4)];assert correct==a['summary'][name][str(st)]['correct']
seal=json.loads((o/'MODEL_SEAL.json').read_text());assert seal['heads']==9 and seal['fit_units']==11 and seal['test_used']==False
# File timestamps are receipts, not a cryptographic chronology certificate.
assert (o/'TRANSFORM_SEAL.json').stat().st_mtime<=(o/'SELECTION_SEAL.json').stat().st_mtime<=(o/'MODEL_SEAL.json').stat().st_mtime<=(o/'pixels_3/INDEX.json').stat().st_mtime<=(o/'predictions.private.pt').stat().st_mtime
for st,n in enumerate([3236,2254,36,586]):
 ix=json.loads((o/f'pixels_{st}/INDEX.json').read_text());ex=json.loads((o/f'pixels_{st}/EXIT.json').read_text());assert ix['rows']==n and sum(c['rows'] for c in ix['chunks'])==n and ex['exit_code']==0
# Coordinator wait4 includes waited reader descendants; do not add child CPU a second time.
co=json.loads((o/'COORDINATOR_EXIT.json').read_text());wr=json.loads((o/'WORKER_RESOURCE_RECEIPT.json').read_text());readers=sum(c['CPU_seconds'] for c in a['reader_costs'])
phases=['preflight','preflight_v2','injected_failure','preflight_final','injected_failure_final','preflight_bridge','injected_failure_bridge','run','run_v2'];cells=[];intervals=[]
for phase in phases:
 d=r/phase;c=json.loads((d/'COORDINATOR_EXIT.json').read_text());end=(d/'COORDINATOR_EXIT.json').stat().st_mtime;intervals.append((end-c['wall_seconds'],end));cells.append(dict(phase=phase,**c))
intervals.sort();merged=[]
for lo,hi in intervals:
 if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
 else:merged.append([lo,hi])
wallunion=sum(hi-lo for lo,hi in merged)
result=dict(pass_all=True,new_methods=3,all_methods=54,all_score_cells=162,exact_inherited_score_cells=153,certified_fit_units=11,artifact_units=9,feature_rows=6112,test_rows=586,test_known_groups=500,finite=True,metrics_replay=True,G_argmax_exact=True,G_max_score_error=max(v for k,v in a['parity'].items() if 'score' in k),source_commit='631df06',seal_order_file_receipts=True,remote_window_bytes=used(r.parent),max_file_bytes=max(p.stat().st_size for p in r.rglob('*') if p.is_file()),reader_self_CPU_seconds=readers,parent_worker_self_CPU_seconds=wr['CPU_seconds'],formal_wait4_includes_descendants_CPU_seconds=co['worker_CPU_seconds'],formal_reader_plus_self_CPU_seconds=readers+wr['CPU_seconds'],coordinator_cells=cells,activity_wall_union_seconds=wallunion,activity_wall_union_basis='coordinator monotonic duration anchored to exit-file mtime; overlapping preflight wall not added',new_known_wait4_CPU_total=sum(c['worker_CPU_seconds'] for c in cells),new_coordinator_CPU_total=sum(c['coordinator_CPU_seconds'] for c in cells),whole_research_CPU_total=None,historical_CPU_gap_retained=True)
(o/'VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
