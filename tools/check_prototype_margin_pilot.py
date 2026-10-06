"""Forward-only final-task pilot; alpha selected without validation labels."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from run_prototype_single import Images,manifests,extract,save
from run_multilabel import ApartFeatures
from diagnose_prototype_scores import summarize
from prototype_margin import heads,class_margin_losses,select_guard


def run(config):
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def budget():
        if time.monotonic()-start>config['max_wall_seconds']:raise TimeoutError('Pilot budget exceeded')
    save(root/'STATUS.json',dict(status='RUNNING',started=time.time(),optimizer_updates=0))
    try:
        seed=config['seed'];torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(config['legacy_repo'],config['weight'],len(config['order']),'cuda:0',seed)
        from run_medical_v2 import transform
        train,val=manifests(config);classes=config['order'][-2:];rows=[r for r in train if r['label'] in classes]
        fit=np.ones(len(rows),dtype=bool)
        for c in classes:
            groups=sorted({r['identity_component'] for r in rows if r['label']==c})
            rng=np.random.default_rng(seed+4*100003+c*2003);rng.shuffle(groups)
            held=set(groups[:max(1,len(groups)//5)])
            for i,r in enumerate(rows):
                if r['label']==c and r['identity_component'] in held:fit[i]=False
        def loader(data):
            return DataLoader(Images(data,config['images'],transform(False),seed+4*100003),batch_size=config['batch_size'],num_workers=4,
                multiprocessing_context='spawn',persistent_workers=True,generator=torch.Generator().manual_seed(seed+4*2003))
        current=loader(rows);path=Path(config['models']['B'])
        previous=torch.load(path/'stage_3.pt',map_location='cpu',weights_only=False);final=torch.load(path/'stage_4.pt',map_location='cpu',weights_only=False)
        encoder.load_state_dict(previous['adapter'],strict=False);encoder.eval();before,y=extract(encoder,current,budget)
        encoder.load_state_dict(final['adapter'],strict=False);after,ay=extract(encoder,current,budget);assert np.array_equal(y,ay)
        seen=final['seen'];old_count=len(previous['seen']);mapped=np.array([seen.index(int(c)) for c in y])
        options=heads(previous['bank'],before[fit],after[fit],y[fit],classes)
        # Fixed global-transport old centers provide a common proxy input for all heads.
        proxy=options[0]['bank']['mu'][:old_count];proxy_labels=np.arange(old_count)
        current_losses=[];old_losses=[]
        for o in options:
            current_losses.append(class_margin_losses(after[~fit]@o['head'],mapped[~fit],list(range(old_count,len(seen))),list(range(old_count))))
            old_losses.append(class_margin_losses(proxy@o['head'],proxy_labels,list(range(old_count)),list(range(len(seen)))))
        selected,eligible=select_guard(current_losses,old_losses)
        selected_alpha=options[selected]['alpha']
        # Lock selection before extracting validation; full current training stats then rebuild the head.
        result=dict(selected_alpha=selected_alpha,eligible=eligible.tolist(),current_calibration_losses=np.asarray(current_losses).tolist(),
            old_proxy_losses=np.asarray(old_losses).tolist(),fit_n=int(fit.sum()),head_calibration_n=int((~fit).sum()),
            encoder_saw_calibration_images_in_original_training=True,old_proxy_is_not_old_image_performance=True,
            selection_used_validation=False,optimizer_updates=0,test_accessed=False,independent_confirmation=False)
        save(root/'SELECTION.json',result)
        full=heads(previous['bank'],before,after,y,classes)
        assert np.allclose(full[-1]['head'],final['head'].numpy(),rtol=1e-5,atol=1e-6)
        z,vy=extract(encoder,loader(val),budget)
        result['conditions']={str(o['alpha']):dict(summary=summarize(z@o['head'],vy,seen),ridge_residual=o['ridge_residual']) for o in full}
        result['selected_summary']=result['conditions'][str(selected_alpha)]['summary']
        expected=json.loads((path/'metrics.json').read_text())
        assert abs(result['conditions']['1.0']['summary']['balanced_accuracy']-expected['final_balanced_accuracy'])<1e-12
        result['fixed_graph_reconstruction_matches']=True
        save(root/'RESULTS.json',result);save(root/'STATUS.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start,ended=time.time(),optimizer_updates=0,validation_n=len(vy)))
        print(json.dumps(dict(selected_alpha=selected_alpha,eligible=eligible.tolist(),BA=result['selected_summary']['balanced_accuracy'])))
    except BaseException as exc:
        save(root/'STATUS.json',dict(status='INCOMPLETE',elapsed_seconds=time.monotonic()-start,error=str(exc),optimizer_updates=0));raise

if __name__=='__main__':run(json.load(sys.stdin))
