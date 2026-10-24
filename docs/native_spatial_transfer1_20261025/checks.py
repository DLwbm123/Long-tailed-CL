"""Once-only full native20image qualification, actual four-arm systems and throughput."""
import io,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS,affine,numpy_affine,pack,unpack_numpy
from data import UnnormalizedImages
from foundation import save,write
from inference import arrays,tokens
from moments import add_current,solve

def qualify(b,base,cfg,p,cost,record,status):
    def check(name,value,passed):record(dict(check=name,value=value,PASS=bool(passed)));assert passed,name
    by={c:sorted([r for r in base if r['label']==c],key=lambda r:(r['group'],r['relative_path'])) for c in cfg['order'][:10]};used=set();selected={}
    for c in sorted(by,key=lambda c:(len({r['group'] for r in by[c]}),c)):
        selected[c]=[]
        for r in by[c]:
            if r['group'] in used:continue
            selected[c].append(r);used.add(r['group'])
            if len(selected[c])==2:break
        check('legal_two_patient_quota',dict(label=c,count=len(selected[c])),len(selected[c])==2)
    rows=sum([selected[c] for c in cfg['order'][:10]],[]);raw=[];zs=[];times=[];table={a:[] for a in ARMS};it=iter(DataLoader(UnnormalizedImages(rows,cfg),batch_size=cfg['microbatch'],shuffle=False,num_workers=0))
    while True:
        tick=time.perf_counter()
        try:x,ix=next(it)
        except StopIteration:break
        with torch.no_grad():
            z=b.forward_features((x.cuda().double()-.5)/.5)
            for a in ARMS:table[a].append(affine(z,a).cpu())
            zs.append(z.cpu());raw.append(x);torch.cuda.synchronize()
        times.append((time.perf_counter()-tick)/len(x))
    raw=torch.cat(raw);zs=torch.cat(zs);table={a:torch.cat(v) for a,v in table.items()};state=arrays(b.state_dict());reference_times=[];chain_error=feature_error=0.
    for i in range(20):
        tick=time.perf_counter();z=tokens(raw[i:i+1].numpy(),state);reference_times.append(time.perf_counter()-tick);chain_error=max(chain_error,float(abs(z-zs[i:i+1].numpy()).max()))
        for a in ARMS:feature_error=max(feature_error,float(abs(numpy_affine(z,a)-table[a][i:i+1].numpy()).max()))
    cost['qualification_native_encodes']=20;cost['qualification_NumPy_full_native_rows']=20;cost['qualification_pool_rows_per_arm']=20;check('full_native_embedding12block_reference',dict(error=chain_error),chain_error<=cfg['full_chain_tolerance']);check('four_real_spatial_operators',dict(error=feature_error),feature_error<=cfg['score_absolute_tolerance'])
    grams={a:torch.zeros(table[a].shape[1],table[a].shape[1],dtype=torch.float64) for a in ARMS};means={a:[] for a in ARMS};solve_times=[];system=None
    for stage,classes in enumerate([cfg['order'][:8],cfg['order'][8:10]]):
        for a in ARMS:add_current(rows,table[a],classes,grams[a],means[a],record)
        tick=time.perf_counter()
        for a in ARMS:
            h,m,rhs=solve(grams[a],means[a],cfg['ridge']);ref=np.linalg.solve(m.numpy(),rhs.numpy());error=float(abs(table[a].numpy()@ref-(table[a]@h).numpy()).max());check('real8plus2_system',dict(stage=stage,arm=a,error=error),error<=cfg['score_absolute_tolerance']);system=dict(matrix=pack(m),arrival=pack(grams[a]),rhs=rhs,head=h,d=len(m))
        solve_times.append(time.perf_counter()-tick);status('qualification_systems',stage=stage)
    save(p/'qualification_system.private.pt',system);loaded=torch.load(p/'qualification_system.private.pt',map_location='cpu',weights_only=False);error=float(abs(unpack_numpy(loaded['matrix'].numpy(),loaded['d'])-m.numpy()).max());check('actual_packed_disk_persistence',dict(error=error),error==0)
    buffer=io.BytesIO();torch.save(b.state_dict(),buffer);nbytes=buffer.tell();buffer.seek(0);loaded=torch.load(buffer,map_location='cpu',weights_only=False);check('original_weight_RAM_roundtrip',dict(bytes=nbytes),all(torch.equal(v.detach().cpu(),loaded[k]) for k,v in b.state_dict().items()));del buffer,loaded
    tick=time.perf_counter()
    for a in ARMS:
        x=table[a].repeat(7,1)[:128];w=torch.full((128,),1/128,dtype=torch.float64);g=x.T@(w[:,None]*x);ref=x.numpy().T@(w[:,None].numpy()*x.numpy());error=float(abs(g.numpy()-ref).max());check('synthetic128row_full_dimension_Gram_throughput',dict(arm=a,error=error,real_new_images=0),error<=1e-9)
    gram_seconds=(time.perf_counter()-tick)/128
    result=dict(state='PASS',real_base_images=20,real_patients=20,optimizer_steps=0,production_seconds_per_image=max(times),NumPy_seconds_per_image=max(reference_times),paired_Gram_seconds_per_row=gram_seconds,paired_system_seconds=max(solve_times),scope='Packed disk persistence and full original-weight RAM roundtrip; no new learned model or optimizer.128row throughput repeats qualified descriptors, not128new images.',science_started=False);write(p/'REAL_QUALIFICATION.json',result);return result
