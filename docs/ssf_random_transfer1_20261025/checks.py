"""Single real qualification: full maps, two SGD updates, endpoint and disk interfaces."""
import io,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS,affine,numpy_affine,pack,unpack_numpy
from data import UnnormalizedImages,patient_draw
from foundation import save,write
from inference import arrays,tokens,cosine_loss
from moments import add_current,solve
from training import optimizer,update

def select_rows(base,cfg,record):
    def check(name,value,passed):record(dict(check=name,value=value,PASS=bool(passed)));assert passed,name
    by={c:sorted([r for r in base if r['label']==c],key=lambda r:(r['group'],r['relative_path'])) for c in cfg['order'][:10]};used=set();selected={}
    for c in sorted(by,key=lambda c:(len({r['group'] for r in by[c]}),c)):
        selected[c]=[]
        for r in by[c]:
            if r['group'] in used:continue
            selected[c].append(r);used.add(r['group'])
            if len(selected[c])==2:break
        check('legal_two_distinct_patient_quota',dict(label=c,count=len(selected[c])),len(selected[c])==2)
    return sum([selected[c] for c in cfg['order'][:10]],[])

def qualify(b,m,W,base,rows,cfg,p,cost,record,status):
    def check(name,value,passed):record(dict(check=name,value=value,PASS=bool(passed)));assert passed,name
    raw=torch.stack([UnnormalizedImages(rows,cfg)[i][0] for i in range(20)]);state=arrays(b.state_dict());init=m.small_state();head=m.head.detach().cpu().clone();ds=UnnormalizedImages(base,cfg);ix,labels=patient_draw(base,cfg['order'][:10],cfg['seed']+90000);draw=None;y=torch.tensor([cfg['order'].index(c) for c in labels]);opt=optimizer(m,cfg);count=dict(updates=0,forward_rows=0,backward_rows=0);step_times=[];cost['qualification_optimizer']=count
    # Two actual discarded updates; the second uses nonzero optimizer momentum.
    for j in range(2):
        reader_tick=time.perf_counter();draw=torch.stack([ds[i][0] for i in ix]);reader_seconds=time.perf_counter()-reader_tick;cost['qualification_draw_pixel_reads']=cost.get('qualification_draw_pixel_reads',0)+len(draw)
        receipt=update(m,opt,draw,y,cfg,count,record,audit=True,step=j);step_times.append(receipt['production_seconds']+reader_seconds);record(dict(check='production_step_timing',step=j,reader_seconds=reader_seconds,SSF_step_seconds=receipt['production_seconds'],independent_operator_seconds=receipt['audit_seconds'],PASS=True));status('qualification_SGD',step=j)
    cost['qualification_optimizer']=count;cost['qualification_operator_audit_steps']=2
    # Specified scalar finite differences on the complete production objective, not all derivatives.
    cost['qualification_FD_forward_rows']=0;cost['qualification_FD_backward_rows']=0
    m.zero_grad(set_to_none=True);cost['qualification_FD_forward_rows']+=1;objective=m.objective(draw[:1].cuda().double(),y[:1].cuda());objective.backward();cost['qualification_FD_backward_rows']+=1;fd=[]
    for name,w in [('patch_scale',m.mods['patch_scale']),('temporary_head',m.head)]:
        old=float(w.flatten()[0]);grad=float(w.grad.flatten()[0]);h=1e-5
        with torch.no_grad():
            w.flatten()[0]=old+h;cost['qualification_FD_forward_rows']+=1;plus=float(m.objective(draw[:1].cuda().double(),y[:1].cuda()));w.flatten()[0]=old-h;cost['qualification_FD_forward_rows']+=1;minus=float(m.objective(draw[:1].cuda().double(),y[:1].cuda()));w.flatten()[0]=old
        error=abs((plus-minus)/(2*h)-grad);fd.append(error);check('complete_objective_scalar_FD',dict(parameter=name,error=error),error<=2e-5)
    m.zero_grad(set_to_none=True);cost['qualification_FD_forward_rows']=5;cost['qualification_FD_backward_rows']=1
    mods=arrays(m.mods.state_dict());table={a:[] for a in ARMS};zs={k:[] for k in ['native','ssf']};times=[]
    # Each timing includes actual reader + both models and all four production readouts.
    it=iter(DataLoader(UnnormalizedImages(rows,cfg),batch_size=1,num_workers=0,shuffle=False))
    while True:
        tick=time.perf_counter()
        try:x,idx=next(it)
        except StopIteration:break
        with torch.no_grad():
            zn=b.forward_features((x.cuda().double()-.5)/.5);zs['native'].append(zn.cpu());za=m.forward_features(x.cuda().double());zs['ssf'].append(za.cpu())
            for a in ARMS:table[a].append(affine(za if a.startswith('SSF') else zn,a,W).cpu())
            torch.cuda.synchronize()
        times.append(time.perf_counter()-tick);cost['qualification_native_encodes']=len(zs['native']);cost['qualification_SSF_encodes']=len(zs['ssf'])
    table={a:torch.cat(v) for a,v in table.items()};zs={k:torch.cat(v) for k,v in zs.items()};rt=[];chain=feature=0.;wn=W.detach().cpu().numpy()
    for i in range(20):
        for k,s in [('native',None),('ssf',mods)]:
            tick=time.perf_counter();z=tokens(raw[i:i+1].numpy(),state,s);rt.append(time.perf_counter()-tick);cost['qualification_NumPy_full_maps']=len(rt);chain=max(chain,float(abs(z-zs[k][i:i+1].numpy()).max()))
            for a in ([ARMS[0],ARMS[2]] if k=='native' else [ARMS[1],ARMS[3]]):feature=max(feature,float(abs(numpy_affine(z,a,wn)-table[a][i:i+1].numpy()).max()))
    check('both_maps_complete_embedding12block_NumPy',dict(error=chain,rows=40),chain<=cfg['full_chain_tolerance']);check('four_readout_NumPy',dict(error=feature),feature<=cfg['score_absolute_tolerance'])
    with torch.no_grad():ez=m.forward_features(draw[:1].cuda().double()).cpu().numpy();actual=float(m.objective(draw[:1].cuda().double(),y[:1].cuda()))
    error=abs(cosine_loss(ez,m.head.detach().cpu().numpy(),y[:1].numpy())-actual);check('cosine_CE_target_NumPy',dict(error=error),error<=1e-9);cost['qualification_target_forward_rows']=2
    grams={a:torch.zeros(table[a].shape[1],table[a].shape[1],dtype=torch.float64) for a in ARMS};means={a:[] for a in ARMS};st=[]
    for stage,classes in enumerate([cfg['order'][:8],cfg['order'][8:10]]):
        for a in ARMS:add_current(rows,table[a],classes,grams[a],means[a],record)
        tick=time.perf_counter()
        for a in ARMS:
            h,mat,rhs=solve(grams[a],means[a],cfg['ridge']);ref=np.linalg.solve(mat.numpy(),rhs.numpy());error=float(abs(table[a].numpy()@ref-(table[a]@h).numpy()).max());check('real8plus2_system',dict(stage=stage,arm=a,error=error),error<=cfg['score_absolute_tolerance'])
        st.append(time.perf_counter()-tick);status('qualification_systems',stage=stage)
    save(p/'qualification_system.private.pt',dict(matrix=pack(mat),arrival=pack(grams[a]),rhs=rhs,head=h,d=len(mat)));loaded=torch.load(p/'qualification_system.private.pt',map_location='cpu',weights_only=False);error=float(abs(unpack_numpy(loaded['matrix'].numpy(),loaded['d'])-mat.numpy()).max());check('actual_max_packed_system_disk',dict(error=error),error==0)
    buffer=io.BytesIO();torch.save(dict(ssf=m.small_state(),projection=W.cpu()),buffer);n=buffer.tell();buffer.seek(0);loaded=torch.load(buffer,map_location='cpu',weights_only=False);check('SSF_RP_RAM_roundtrip',dict(bytes=n),all(torch.equal(v,loaded['ssf'][k]) for k,v in m.small_state().items()) and torch.equal(W.cpu(),loaded['projection']));del buffer,loaded
    tick=time.perf_counter()
    for a in ARMS:
        x=table[a].repeat(7,1)[:128];w=torch.full((128,),1/128,dtype=torch.float64);g=x.T@(w[:,None]*x);ref=x.numpy().T@(w[:,None].numpy()*x.numpy());error=float(abs(g.numpy()-ref).max());check('repeated128row_Gram_throughput',dict(arm=a,error=error,new_images=0),error<=1e-9)
    gram=(time.perf_counter()-tick)/128
    m.load_small(init)
    with torch.no_grad():m.head.copy_(head.to(m.head.device))
    cost.update(qualification_native_encodes=20,qualification_SSF_encodes=20,qualification_NumPy_full_maps=40,qualification_SGD_audit_seconds_excluded_from_step_only=True)
    q=dict(state='PASS',real_images=20,distinct_patients=20,discarded_optimizer_steps=2,production_step_seconds=max(step_times),production_seconds_per_image=max(times),NumPy_seconds_per_image=max(rt),paired_Gram_seconds_per_row=gram,paired_system_seconds=max(st),science_started=False,scope='Full both-map references;2scalarFD notallparameterderivatives;SSF/RPRAM+maxsystemdisk.128throughput repeatsqualifieddescriptors. Updates discarded and reset.')
    write(p/'REAL_QUALIFICATION.json',q);return q
