"""One true base-image qualification; any assertion closes this fixed recipe."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from core import NativeUpper,affine_descriptor,redundancy_loss
from data import UnnormalizedImages
from foundation import save,write
from training import ARMS,Learner,optimizer,update,canonical_outputs
from inference import body_state,numpy_embedding,numpy_affine
from native_reference import np_tokens
from reference import objective_and_grad
from geometry import augment,view_receipt
from numpy_objective import objective
from coupled import backward_coupled
from moments import add_current,solve


def qualify(backbone,native,base,cfg,p,cost,record,status,encode,paired):
    def check(name,value,passed):
        record(dict(check=name,value=value,PASS=bool(passed)))
        assert passed,name
    # First Torch execution of the new reference criterion, not a rerun of finite differences.
    x=torch.tensor([[[.3*np.sin(1+i*3+k*2+j) for j in range(3)] for k in range(2)] for i in range(5)],dtype=torch.float64,device='cuda',requires_grad=True)
    y=torch.tensor([[[.4*np.cos(2+i*2+k+j*3) for j in range(3)] for k in range(2)] for i in range(5)],dtype=torch.float64,device='cuda',requires_grad=True)
    expected,gradient=objective_and_grad(x.detach().cpu().tolist(),y.detach().cpu().tolist());value=redundancy_loss(x,y);value.backward()
    error=abs(float(value.detach())-expected);grad_error=max(float(abs(x.grad.cpu().numpy()-np.asarray(gradient[0])).max()),float(abs(y.grad.cpu().numpy()-np.asarray(gradient[1])).max()))
    cost['Torch_math_objective_calls']=1;cost['Torch_math_backward_calls']=1;cost['new_reference_objective_calls']=1
    check('Torch_new_objective_against_independent_derivatives',dict(error=error,derivative_error=grad_error),error<=cfg['math_objective_tolerance'] and grad_error<=cfg['math_derivative_tolerance'])
    rows=[]
    for c in cfg['order'][:10]:
        candidates=sorted((r for r in base if r['label']==c),key=lambda r:(r['group'],r['relative_path']));patients=set()
        for row in candidates:
            if row['group'] not in patients:rows.append(row);patients.add(row['group'])
            if len(patients)==2:break
        assert len(patients)==2
    data=UnnormalizedImages(rows,cfg);pixels=torch.stack([data[i][0] for i in range(20)])
    tokens,embeddings,lowers=encode((pixels-.5)/.5,'qualification_canonical_encodes',True)
    original=[]
    with torch.no_grad():
        for i in range(0,20,cfg['microbatch']):original.append(affine_descriptor(native(tokens[i:i+cfg['microbatch']].cuda().double())).cpu())
    original=torch.cat(original)
    means=[]
    for c in cfg['order'][:10]:means.append(original[[i for i,r in enumerate(rows) if r['label']==c],:1536].mean(0)*np.sqrt(2))
    means=torch.stack(means);cost['qualification_native_feature_rows']=20
    cpu=None
    from foundation import build_backbone
    cpu=build_backbone(cfg,'cpu');reference_lower=body_state(cpu,0,8,False);reference_upper=body_state(cpu,8,12)
    embed=numpy_embedding(((pixels-.5)/.5).numpy(),cpu);embedding_error=float(abs(embed-embeddings.double().numpy()).max())
    # Compare quantized patch input to its expected rounding, with no claim of equality to unquantized pixels.
    with torch.no_grad():gpu_embedding=__import__('foundation').prefix(backbone,((pixels-.5)/.5).cuda().double()).cpu().numpy()
    cost['qualification_extra_embedding_rows']=20
    raw_embed_error=float(abs(embed-gpu_embedding).max());check('real_reader_and_patch_embedding',dict(raw_FP64_error=raw_embed_error,FP16_rounding_max=embedding_error),raw_embed_error<=cfg['full_chain_tolerance'])
    lower_error=0.;upper_error=0.
    for i in range(20):
        np_lower=np_tokens(embeddings[i:i+1].numpy(),reference_lower,apply_norm=False);lower_error=max(lower_error,float(abs(np_lower-lowers[i:i+1].numpy()).max()))
        np_feature=numpy_affine(tokens[i:i+1].numpy(),reference_upper);upper_error=max(upper_error,float(abs(np_feature-original[i:i+1].numpy()).max()))
    cost['qualification_NumPy_lower_rows']=20;cost['qualification_NumPy_native_upper_rows']=20
    check('real_full_native_8plus4_chain',dict(lower_error=lower_error,native_upper_error=upper_error),lower_error<=cfg['full_chain_tolerance'] and upper_error<=cfg['full_chain_tolerance'])
    receipt=view_receipt(cfg['seed']+991,20);transformed=augment(pixels,receipt);pixel_error=0.
    padded=np.pad(pixels.numpy(),((0,0),(0,0),(16,16),(16,16)),mode='reflect')
    for k,(dy,dx) in enumerate(receipt['shifts']):
        params=np.asarray(receipt['photometric'][k],dtype=np.float32);crop=padded[:,:,16-16*dy:240-16*dy,16-16*dx:240-16*dx]
        expected=((np.clip(crop*params[:,0,None,None,None]+params[:,1,None,None,None],0,1)-np.float32(.5))/np.float32(.5))
        pixel_error=max(pixel_error,float(abs(expected-transformed[k].numpy()).max()))
    check('real_view_pixel_transform',dict(error=pixel_error,matched_cells=len(receipt['first_cells'])),pixel_error<=cfg['full_chain_tolerance'])
    save(p/'qualification_inputs.private.pt',dict(rows=rows,pixels=pixels,embeddings=embeddings,lowers_FP64=lowers,lower=tokens))
    models={}
    for a in ARMS[1:]:torch.manual_seed(cfg['seed']);models[a]=Learner(NativeUpper(backbone),a,means).cuda().double()
    opts={a:optimizer(m,cfg) for a,m in models.items()};paired_times=[]
    for step in range(2):
        torch.cuda.synchronize();tick=time.perf_counter();inputs,labels,receipt,ids=paired(rows,cfg['seed']+700+step,'qualification_augmented_encodes')
        audit_seconds=0.
        for a,m in models.items():
            detail=update(m,opts[a],inputs,labels,receipt,cfg,cost['qualification_arms'][a],lambda v:record(dict(arm=a,qualification_step=step,**v)),audit=True)
            audit_seconds+=detail['qualification_audit_seconds']
        torch.cuda.synchronize();whole=time.perf_counter()-tick;paired_times.append(whole-audit_seconds)
        record(dict(check='qualification_benchmark_accounting',whole_paired_seconds=whole,qualification_only_operator_reference_seconds=audit_seconds,formal_canonical_paired_seconds=whole-audit_seconds));status('real_qualification_steps',completed=step+1)
    from coupled import backward_coupled
    for a,m in models.items():
        opts[a].zero_grad(set_to_none=True)
        backward_coupled(m.upper,inputs,lambda out:m.objective(out,labels,receipt),lambda v:record(dict(arm=a,qualification_probe=True,**v)),cfg['microbatch'])
        cost['qualification_gradient_probe_forward_rows']=cost.get('qualification_gradient_probe_forward_rows',0)+128
        cost['qualification_gradient_probe_backward_rows']=cost.get('qualification_gradient_probe_backward_rows',0)+64
        derivative_error=0.
        for name,index in [('blocks.0.attn.qkv.weight',(0,0)),('norm.weight',(0,))]:
            parameter=dict(m.upper.named_parameters())[name];analytic=float(parameter.grad[index]);old=float(parameter[index]);eps=1e-5
            with torch.no_grad():parameter[index]=old+eps
            plus=float(m.objective(canonical_outputs(m,inputs,cfg['microbatch']),labels,receipt))
            with torch.no_grad():parameter[index]=old-eps
            minus=float(m.objective(canonical_outputs(m,inputs,cfg['microbatch']),labels,receipt))
            with torch.no_grad():parameter[index]=old
            derivative_error=max(derivative_error,abs((plus-minus)/(2*eps)-analytic))
            cost['qualification_finite_difference_forward_rows']=cost.get('qualification_finite_difference_forward_rows',0)+128
        check('actual_native_upper_coupled_replay_derivative',dict(arm=a,selected_parameters=2,error=derivative_error),derivative_error<=cfg['math_derivative_tolerance'])
        out=canonical_outputs(m,inputs,cfg['microbatch']);native_state={k:v.detach().cpu().numpy() for k,v in m.upper.state_dict().items()};proj={k:v.detach().cpu().numpy() for k,v in m.projectors.state_dict().items()}
        expected=objective(out.detach().cpu().numpy(),labels.numpy(),receipt,proj,m.head.detach().cpu().numpy(),a);actual=float(m.objective(out,labels,receipt));check('real_full_coupled_objective_NumPy',dict(arm=a,error=abs(expected-actual)),abs(expected-actual)<=cfg['score_absolute_tolerance'])
        cost['qualification_objective_GPU_rows']=cost.get('qualification_objective_GPU_rows',0)+64;cost['qualification_objective_NumPy_rows']=cost.get('qualification_objective_NumPy_rows',0)+64
        path=p/f'qualification_{a}.private.pt';save(path,dict(upper=m.upper.state_dict(),projectors=m.projectors.state_dict(),head=m.head))
        loaded=torch.load(path,map_location='cpu',weights_only=False);check('real_save_load',dict(arm=a),all(torch.equal(v.detach().cpu(),loaded['upper'][k]) for k,v in m.upper.state_dict().items()))
        delta=max(float((v.detach().cpu()-native.state_dict()[k].cpu()).abs().max()) for k,v in m.upper.state_dict().items());check('real_native_upper_changed',dict(arm=a,max_delta=delta),delta>0)
        features=[];reference=[]
        with torch.no_grad():
            for i in range(0,20,cfg['microbatch']):features.append(affine_descriptor(m.upper(tokens[i:i+cfg['microbatch']].cuda().double())).cpu())
        features=torch.cat(features)
        for i in range(20):reference.append(numpy_affine(tokens[i:i+1].numpy(),native_state))
        error=float(abs(features.numpy()-np.concatenate(reference)).max());check('real_trained_full_upper4_NumPy',dict(arm=a,error=error),error<=cfg['full_chain_tolerance'])
        cost['qualification_trained_upper_GPU_rows']=cost.get('qualification_trained_upper_GPU_rows',0)+20;cost['qualification_trained_upper_NumPy_rows']=cost.get('qualification_trained_upper_NumPy_rows',0)+20
        gram=torch.zeros(1537,1537,dtype=torch.float64);means=[]
        for stage,classes in enumerate([cfg['order'][:8],cfg['order'][8:10]]):
            add_current(rows,features,classes,gram,means,lambda v:record(dict(arm=a,qualification_readout=True,**v)));h,mat,rhs=solve(gram,means,cfg['ridge']);expected=np.linalg.solve(mat.numpy(),rhs.numpy());score_error=float(abs(features.numpy()@expected-(features@h).numpy()).max());check('real_arrival_full_system',dict(arm=a,stage=stage,error=score_error,columns=len(means)),score_error<=cfg['score_absolute_tolerance'])
    check('original_native_weights_frozen',{},all(torch.equal(v.detach().cpu(),cpu.state_dict()[k]) for k,v in backbone.state_dict().items()))
    result=dict(state='PASS',paired_step_seconds=max(paired_times),paired_step_seconds_samples=paired_times,unique_legal_base_images=20,discarded_updates=6,science_started=False,scope='Full native forward interfaces and canonical coupled objective; two real dense native updates per arm, selected full-model derivatives, same-gradient full AdamW. No cross-batchshape trajectory claim.')
    write(p/'REAL_QUALIFICATION.json',result);del models,opts,cpu
    torch.cuda.empty_cache();return result
