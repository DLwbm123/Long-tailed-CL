"""First real qualification on20legal base images; numeric gates persist before assert."""
import io,time,json,copy
import numpy as np
import torch
from core import ARMS,Model,descriptor,mask_receipt
from data import UnnormalizedImages,patient_draw
from training import optimizer,update,counter
from inference import arrays,tokens,affine,objective
from moments import add_current,solve
from foundation import write

def qualify(backbone,base,cfg,p,cost,record,status):
    def check(name,value,passed):record(dict(check=name,value=value,PASS=bool(passed)));assert passed,name
    rows=[]
    for c in cfg['order'][:10]:
        patients=set()
        for r in sorted([r for r in base if r['label']==c],key=lambda r:(r['group'],r['relative_path'])):
            if r['group'] not in patients:rows.append(r);patients.add(r['group'])
            if len(patients)==2:break
        assert len(patients)==2
    ds=UnnormalizedImages(rows,cfg);raw=torch.stack([ds[i][0] for i in range(20)]);initial=arrays(backbone.state_dict());native=[];native_error=0.
    backbone.cuda()
    with torch.no_grad():
        for i in range(20):
            b=backbone;x=b.norm_pre(b.patch_drop(b._pos_embed(b.patch_embed((raw[i:i+1].cuda().double()-.5)/.5))))
            z=b.norm(b.blocks(x));native.append(descriptor(z).cpu());n=tokens(raw[i:i+1].numpy(),initial);native_error=max(native_error,float(abs(n-z.cpu().numpy()).max()))
    cost['qualification_canonical_encodes']=20;cost['qualification_native_numpy_rows']=20
    check('real_full_native_patch_and12block_NumPy',dict(error=native_error),native_error<=cfg['full_chain_tolerance']);native=torch.cat(native);backbone.cpu()
    means=torch.stack([native[[i for i,r in enumerate(rows) if r['label']==c],:1536].mean(0)*np.sqrt(2) for c in cfg['order'][:10]])
    models={}
    for a in ARMS[1:]:torch.manual_seed(cfg['seed']);models[a]=Model(backbone,means).eval()
    paired_times=[]
    for phase in ['A','B']:
        opts={a:optimizer(m,cfg) for a,m in models.items()};tick=time.perf_counter();ids,labels=patient_draw(rows,cfg['order'][:10],cfg['seed']+(700 if phase=='A' else 701));images=torch.stack([ds[i][0] for i in ids]);y=torch.tensor([cfg['order'][:10].index(c) for c in labels]);receipt=mask_receipt(cfg['seed']+(900 if phase=='A' else 901));audit=0.
        for a,m in models.items():
            detail=update(m,opts[a],images,y,receipt,a,phase,cfg,cost['qualification_arms'][a],lambda v:record(dict(arm=a,phase=phase,**v)),audit=True);audit+=detail['operator_audit_seconds']
        elapsed=time.perf_counter()-tick;paired_times.append(elapsed-audit);record(dict(check='qualification_timing',phase=phase,whole_seconds=elapsed,operator_reference_seconds=audit,canonical_paired_seconds=elapsed-audit));status('qualification_steps',phase_done=phase);del opts
    for a,m in models.items():
        m.cuda();receipt=mask_receipt(cfg['seed']+999);image=raw[:1].cuda().double();label=torch.tensor([cfg['order'][:10].index(rows[0]['label'])],device='cuda');phase='A'
        m.zero_grad(set_to_none=True);loss=m.objective(image,label,receipt,a,phase);loss.backward();cost['qualification_probe_backward_rows']=cost.get('qualification_probe_backward_rows',0)+1;derivative=0.
        for name,ix in [('backbone.norm.weight',(0,)),('backbone.patch_embed.proj.weight',(0,0,0,0))]:
            par=dict(m.named_parameters())[name];g=float(par.grad[ix]);old=float(par[ix]);eps=1e-5
            with torch.no_grad():par[ix]=old+eps;plus=float(m.objective(image,label,receipt,a,phase));par[ix]=old-eps;minus=float(m.objective(image,label,receipt,a,phase));par[ix]=old
            derivative=max(derivative,abs((plus-minus)/(2*eps)-g));cost['qualification_FD_forward_rows']=cost.get('qualification_FD_forward_rows',0)+2
        check('real_selected_full_model_derivatives',dict(arm=a,error=derivative,parameters=2,rows=1),derivative<=2e-6);m.zero_grad(set_to_none=True)
        with torch.no_grad():
            masked=a=='MASKED_CONTEXT_THEN_SUPERVISED';z=m(image,receipt,masked);state=arrays(m.backbone.state_dict());np_z=tokens(raw[:1].numpy(),state,receipt,masked);err=float(abs(z.cpu().numpy()-np_z).max());expected=objective(np_z,raw[:1].numpy(),label.cpu().numpy(),receipt,arrays(m.decoder.state_dict()),m.head.detach().cpu().numpy(),a,phase);objerr=abs(float(m.objective(image,label,receipt,a,phase))-expected)
            cost['qualification_masked_model_reference_rows']=cost.get('qualification_masked_model_reference_rows',0)+1;check('actual_pretext_or_CE_NumPy',dict(arm=a,token_error=err,objective_error=objerr),err<=cfg['full_chain_tolerance'] and objerr<=cfg['score_absolute_tolerance'])
            if masked:
                altered=image.clone();view=altered.reshape(1,3,14,16,14,16).permute(0,2,4,1,3,5).reshape(1,196,768).clone();view[:,receipt['tokens']]+=.01;altered=view.reshape(1,14,14,3,16,16).permute(0,3,1,4,2,5).reshape(1,3,224,224);leak=float(abs(m(altered,receipt,True)-z).max());check('masked_pixels_do_not_leak_to_input',dict(error=leak),leak<=1e-10);cost['qualification_mask_leak_rows']=2
            feat=torch.cat([descriptor(m(raw[i:i+1].cuda().double())).cpu() for i in range(20)])
        error=0.
        for i in range(20):error=max(error,float(abs(affine(tokens(raw[i:i+1].numpy(),state))-feat[i:i+1].numpy()).max()))
        cost['qualification_trained_numpy_rows']=cost.get('qualification_trained_numpy_rows',0)+20;cost['qualification_trained_GPU_rows']=cost.get('qualification_trained_GPU_rows',0)+20;check('real_trained_full12_chain',dict(arm=a,error=error),error<=cfg['full_chain_tolerance'])
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);serialized_bytes=buffer.tell();buffer.seek(0);loaded=torch.load(buffer,map_location='cpu',weights_only=False);same=all(torch.equal(v.detach().cpu(),loaded[k]) for k,v in m.state_dict().items());check('RAM_serialization_roundtrip_not_disk',dict(arm=a,bytes=serialized_bytes),same);del buffer,loaded
        changed=max(float(abs(v.detach().cpu()-backbone.state_dict()[k]).max()) for k,v in m.backbone.state_dict().items());check('full_model_actually_updated',dict(arm=a,max_delta=changed),changed>0)
        gram=torch.zeros(1537,1537,dtype=torch.float64);means=[]
        for stage,classes in enumerate([cfg['order'][:8],cfg['order'][8:10]]):
            add_current(rows,feat,classes,gram,means,record);head,mat,rhs=solve(gram,means,cfg['ridge']);ind=np.linalg.solve(mat.numpy(),rhs.numpy());error=float(abs(feat.numpy()@ind-(feat@head).numpy()).max());check('actual8plus2_full_system',dict(arm=a,stage=stage,error=error),error<=cfg['score_absolute_tolerance'])
        m.cpu();status('qualification_arm_done',arm=a)
    check('native_original_unchanged',{},all(np.array_equal(v,backbone.state_dict()[k].numpy()) for k,v in initial.items()))
    result=dict(state='PASS',unique_base_images=20,discarded_optimizer_updates=6,paired_step_seconds=max(paired_times),paired_samples=paired_times,science_started=False,RAM_serialization_not_disk=True);write(p/'REAL_QUALIFICATION.json',result);del models;torch.cuda.empty_cache();return result
