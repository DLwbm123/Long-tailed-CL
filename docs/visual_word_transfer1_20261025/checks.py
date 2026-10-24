"""Once-only real100patient qualification of full production word geometry and model."""
import io,time
import numpy as np
import torch
from core import ARMS,Model,descriptor,shuffle_receipt,corrupt
from data import UnnormalizedImages,patient_draw
from word_data import patient_means,discover,word_draw,render
import numpy_word
from training import optimizer,update
from inference import arrays,tokens,affine
from moments import add_current,solve
from foundation import write

def qualify(backbone,base,cfg,p,cost,record,status):
    def check(name,value,passed):record(dict(check=name,value=value,PASS=bool(passed)));assert passed,name
    rows=[];used=set()
    for c in cfg['order'][:10]:
        chosen=[]
        for r in sorted([r for r in base if r['label']==c],key=lambda r:(r['group'],r['relative_path'])):
            if r['group'] in used:continue
            chosen.append(r);used.add(r['group'])
            if len(chosen)==10:break
        check('qualification_patient_quota',dict(label=c,patients=len(chosen)),len(chosen)==10);rows+=chosen
    ds=UnnormalizedImages(rows,cfg);raw=torch.stack([ds[i][0] for i in range(100)]);original=arrays(backbone.state_dict());native=[];error=0.;backbone.cuda()
    with torch.no_grad():
        for i in range(100):
            z=backbone.forward_features((raw[i:i+1].cuda().double()-.5)/.5);native.append(descriptor(z).cpu());reference=tokens(raw[i:i+1].numpy(),original);error=max(error,float(abs(z.cpu().numpy()-reference).max()))
    cost['qualification_native_GPU_rows']=100;cost['qualification_native_numpy_rows']=100;check('real_full12_native_NumPy',dict(error=error),error<=cfg['full_chain_tolerance']);native=torch.cat(native);backbone.cpu();groups,means=patient_means(rows,native,record);words=discover(means,cfg['seed']+1,record)
    initial_head=torch.stack([native[[i for i,r in enumerate(rows) if r['label']==c],:1536].mean(0)*np.sqrt(2) for c in cfg['order'][:10]]);models={}
    for a in ARMS[1:]:torch.manual_seed(cfg['seed']);models[a]=Model(backbone,initial_head).eval()
    opts={a:optimizer(m,cfg) for a,m in models.items()};paired=[]
    for step in [1,2]:
        tick=time.perf_counter();receipt=word_draw(words,groups,cfg['seed']+1000+step);target=render(ds,receipt);shuffle=shuffle_receipt(cfg['seed']+2000+step);inputs=corrupt(target,shuffle);labels=torch.tensor([r['word'] for r in receipt]);ids,classes=patient_draw(rows,cfg['order'][:10],cfg['seed']+3000+step);sup=torch.stack([ds[i][0] for i in ids]);sy=torch.tensor([cfg['order'][:10].index(c) for c in classes]);audit=0.
        for a,m in models.items():
            x,t,y=(sup,sup,sy) if a==ARMS[1] else (inputs,target,labels);d=update(m,opts[a],x,t,y,a,cfg,cost['qualification_arms'][a],lambda v:record(dict(arm=a,step=step,**v)),audit=True);audit+=d['operator_audit_seconds']
        whole=time.perf_counter()-tick;paired.append(whole-audit);record(dict(check='paired_timing',step=step,whole_seconds=whole,operator_reference_seconds=audit,canonical_paired_seconds=whole-audit));status('qualification_updates',step=step)
    del opts
    render_error=0.
    for i,r in enumerate(receipt):render_error=max(render_error,float(abs(numpy_word.crop(raw[r['index']:r['index']+1].numpy(),r['top'],r['left'])-target[i:i+1].numpy()).max()))
    pixel_error=float(abs(numpy_word.corrupt(target.numpy(),shuffle)-inputs.numpy()).max());check('real_crop_and_RGB_shuffle_reference',dict(crop_error=render_error,shuffle_error=pixel_error),render_error<=2e-7 and pixel_error==0)
    for a,m in models.items():
        m.cuda();x,t,y=(sup[:1],sup[:1],sy[:1]) if a==ARMS[1] else (inputs[:1],target[:1],labels[:1]);x=x.cuda().double();t=t.cuda().double();y=y.cuda();m.zero_grad(set_to_none=True);loss=m.objective(x,t,y,a);loss.backward();cost['qualification_derivative_probe_rows']=cost.get('qualification_derivative_probe_rows',0)+1;error=0.
        for name,ix in [('backbone.norm.weight',(0,)),('backbone.patch_embed.proj.weight',(0,0,0,0))]:
            par=dict(m.named_parameters())[name];g=float(par.grad[ix]);old=float(par[ix]);eps=1e-5
            with torch.no_grad():par[ix]=old+eps;plus=float(m.objective(x,t,y,a));par[ix]=old-eps;minus=float(m.objective(x,t,y,a));par[ix]=old
            error=max(error,abs((plus-minus)/(2*eps)-g));cost['qualification_FD_forward_rows']=cost.get('qualification_FD_forward_rows',0)+2
        check('real_selected_derivatives',dict(arm=a,parameters=2,error=error),error<=2e-6);m.zero_grad(set_to_none=True)
        with torch.no_grad():
            state=arrays(m.backbone.state_dict());z=m(x);nz=tokens(x.cpu().numpy(),state);tokenerror=float(abs(z.cpu().numpy()-nz).max());value=numpy_word.objective(nz,t.cpu().numpy(),y.cpu().numpy(),a,arrays(m.decoder.state_dict()),arrays(m.word_head.state_dict()),m.disease_head.detach().cpu().numpy());objerror=abs(value-float(m.objective(x,t,y,a)));check('real_word_or_supervised_objective_NumPy',dict(arm=a,token_error=tokenerror,objective_error=objerror),tokenerror<=cfg['full_chain_tolerance'] and objerror<=cfg['score_absolute_tolerance']);features=torch.cat([descriptor(m(raw[i:i+1].cuda().double())).cpu() for i in range(100)])
        error=0.
        for i in range(100):error=max(error,float(abs(affine(tokens(raw[i:i+1].numpy(),state))-features[i:i+1].numpy()).max()))
        cost['qualification_trained_GPU_rows']=cost.get('qualification_trained_GPU_rows',0)+100;cost['qualification_trained_numpy_rows']=cost.get('qualification_trained_numpy_rows',0)+100;check('real_trained_full12_chain',dict(arm=a,error=error),error<=cfg['full_chain_tolerance'])
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);size=buffer.tell();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False);same=all(torch.equal(v.detach().cpu(),saved[k]) for k,v in m.state_dict().items());check('RAM_serialization_only',dict(arm=a,bytes=size),same);del buffer,saved
        delta=max(float(abs(v.detach().cpu()-backbone.state_dict()[k]).max()) for k,v in m.backbone.state_dict().items());check('actual_model_update',dict(arm=a,delta=delta),delta>0)
        gram=torch.zeros(1537,1537,dtype=torch.float64);means=[]
        for stage,classes in enumerate([cfg['order'][:8],cfg['order'][8:10]]):
            add_current(rows,features,classes,gram,means,record);head,mat,rhs=solve(gram,means,cfg['ridge']);ind=np.linalg.solve(mat.numpy(),rhs.numpy());error=float(abs(features.numpy()@ind-(features@head).numpy()).max());check('real8plus2_system',dict(arm=a,stage=stage,error=error),error<=cfg['score_absolute_tolerance'])
        m.cpu();status('qualification_arm_done',arm=a)
    check('native_original_fixed',{},all(np.array_equal(v,backbone.state_dict()[k].numpy()) for k,v in original.items()));out=dict(state='PASS',base_images=100,distinct_patients=100,cohort_C=100,cohort_K=64,discarded_updates=6,paired_samples=paired,paired_step_seconds=max(paired),RAM_serialization_not_disk=True,science_started=False);write(p/'REAL_QUALIFICATION.json',out);del models;torch.cuda.empty_cache();return out
