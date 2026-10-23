"""Single real qualification; every measured gate persisted before assertion."""
import copy,time
import numpy as np
import torch
from core import Model,ArrivalFrame,ARMS
from foundation import save,write
from training import optimizer,update,counter
from inference import body_state,native_features,numpy_scores,scores,restore,frozen_parameter_delta,numpy_embedding
from reference import np_tokens


def make_models(b,cfg):
    models={};initial=None
    for arm in ARMS[1:]:
        torch.manual_seed(cfg['seed']);m=Model(b,arm).to(next(b.parameters()))
        if initial is None:initial=m.adapter_state()
        else:
            with torch.no_grad():
                for k,v in m.named_parameters():
                    if k in initial:v.copy_(initial[k].to(v))
        models[arm]=m
    return models


def qualify(b,cpu,tokens,means,labels,proof,image_proof,native_output,cfg,p,cost,checkpoint):
    receipts=[]
    def record(v):receipts.append(v);write(p/'QUALIFICATION_PROGRESS.json',dict(receipts=receipts,cost=cost));checkpoint('real_GPU_qualification')
    lower=body_state(cpu,0,8);upper=body_state(cpu,8,12)
    embedding_error=float(abs(numpy_embedding(image_proof['images'].numpy(),cpu)-image_proof['raw_embedding'].numpy()).max())
    record(dict(check='reader_patch_embedding',error=embedding_error,rows=len(image_proof['images'])));assert embedding_error<=cfg['score_absolute_tolerance']
    lower_error=0.;upper_error=0.;native=native_output.numpy()
    from inference import numpy_features
    for i,v in enumerate(proof):
        ref=np_tokens(v['embedding'][None].numpy(),lower,apply_norm=False)
        lower_error=max(lower_error,float(abs(ref-v['lower'][None].numpy()).max()))
        z=numpy_features(tokens[i:i+1].numpy(),upper)
        upper_error=max(upper_error,float(abs(z-native[i:i+1]).max()))
        cost['qualification_full_chain_rows']+=1
        record(dict(check='real_full8plus4',completed=i+1,lower8_error=lower_error,native_upper4_error=upper_error))
        assert lower_error<=cfg['score_absolute_tolerance'] and upper_error<=cfg['score_absolute_tolerance']
    frame=ArrivalFrame(record);frame.append(means[:8],labels[:8]);models=make_models(b,cfg);timings=[0.,0.];arm_receipts={}
    native_two=native_features(b,tokens[:2],1);cost['qualification_native_probe_rows']+=2
    for arm,m in models.items():
        m.arrive(means[:8],frame.rows,labels[:8])
        with torch.no_grad():zero=torch.cat([m.features(tokens[i:i+1].to(next(m.parameters()))).cpu() for i in range(2)])
        zero_error=float(abs(zero-native_two).max());cost['qualification_zero_B_rows']+=2
        record(dict(check='real_zero_B_bypass',arm=arm,error=zero_error,rows=2));assert zero_error<=cfg['score_absolute_tolerance']
    first_rows=frame.rows.clone()
    for stage,n in enumerate([8,10]):
        if stage:
            frame.append(means[8:],labels[8:]);r=frame.receipt();r.update(check='qualified_frame_extension',old_delta=float(abs(frame.rows[:8]-first_rows).max()));record(r);assert r['Gram_error']<=1e-10 and r['old_delta']==0
            for m in models.values():m.arrive(means[8:],frame.rows,labels[8:])
        for arm,m in models.items():
            cc=cost['qualification_arms'][arm];opt=optimizer(m,cfg)
            ix=torch.arange(64)%(2*n);y=torch.arange(n).repeat_interleave(2)[ix]
            old=m.adapter_state();audit={}
            try:
                details=update(m,opt,tokens[ix],y,cfg,cc,record,audit)
            finally:record(dict(check='completed_update_counts',arm=arm,stage=stage,counts=dict(cc)))
            audit.update(check='same_gradient_AdamW',arm=arm,stage=stage);record(audit);assert audit['normalized_error']<=1
            timings[stage]+=details['canonical_step_seconds']
            current=m.adapter_state();delta=max(float((v-current[k]).abs().max()) for k,v in old.items() if k.endswith(('.qA','.qB','.vA','.vB')))
            frozen=frozen_parameter_delta(m,b);record(dict(check='parameter_permissions',arm=arm,stage=stage,adapter_delta=delta,frozen_delta=frozen));assert delta>0 and frozen==0
            if arm!='LEARNED_CE':
                hd=float(abs(current['head']-frame.rows).max());record(dict(check='fixed_target_head',arm=arm,head_delta=hd));assert hd==0
            save(p/f'qualification_{arm}_{stage}.private.pt',dict(adapter=current,labels=m.labels))
            loaded=torch.load(p/f'qualification_{arm}_{stage}.private.pt',map_location='cpu',weights_only=False)
            cm=restore(cpu,arm,loaded['adapter'],loaded['labels']);ref=scores(cm,tokens[:2],1).numpy();cost['qualification_CPU_score_rows']+=2
            npv=numpy_scores(tokens[:2].numpy(),upper,{k:v.numpy() for k,v in loaded['adapter'].items()});cost['qualification_NumPy_score_rows']+=2
            gpu=scores(m,tokens[:2],1).numpy();cost['qualification_GPU_score_rows']+=2
            er=float(abs(ref-npv).max());ge=float(abs(ref-gpu).max());record(dict(check='real_save_load_readouts',arm=arm,stage=stage,CPU_NumPy_error=er,GPU_CPU_error=ge,rows=2));assert er<=cfg['score_absolute_tolerance'] and ge<=cfg['score_absolute_tolerance']
            arm_receipts[arm]=dict(CPU_NumPy_error=er,GPU_CPU_error=ge,adapter_delta=delta,frozen_delta=frozen)
            del cm,opt;torch.cuda.empty_cache()
    result=dict(arms=arm_receipts,receipts=receipts,paired_stage_step_seconds=timings,paired_step_seconds=max(timings),discarded_updates=sum(v['updates'] for v in cost['qualification_arms'].values()),embedding_error=embedding_error,full_lower_error=lower_error,full_native_upper_error=upper_error)
    write(p/'REAL_QUALIFICATION.json',result)
    del models;torch.cuda.empty_cache();return result
