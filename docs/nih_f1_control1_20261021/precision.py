"""Source-prescribed precision provenance test, never a model-selection sweep."""
import json,time,sys,random
from pathlib import Path
import torch,numpy as np
from common import settings,write

def main():
    cfg,out=settings();start=time.time();encodes=0;records=[]
    original=json.loads(Path(cfg['source_input']).read_text());sys.path.insert(0,str(Path(cfg['source'])/'tools'));sys.path.insert(1,str(Path(original['legacy_repo'])/'third_party/APART'))
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from run_prototype_single import Images,manifests
    from torch.utils.data import DataLoader
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(4)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is True
    source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    model=ApartFeatures(original['legacy_repo'],original['weight'],20,'cuda:0',cfg['seed']);model.load_state_dict(source['model'],strict=True);model.requires_grad_(False).eval()
    flags=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
    assert flags['matmul_tf32'] is False and flags['cudnn_tf32'] is True
    _,val=manifests(original);seen=source['seen'];rows=[r for r in val if r['label'] in seen];assert len(rows)==150
    w=source['head'].numpy();features=[];predictions=[];score_sets=[]
    try:
        for setting in [True,False]:
            torch.backends.cudnn.allow_tf32=setting
            loader=DataLoader(Images(rows,cfg['images'],transform(False),cfg['seed']+100003),batch_size=64,num_workers=0,shuffle=False)
            xs=[];ys=[]
            for images,labels in loader:
                assert time.time()<cfg['hard_deadline_epoch']-60 and torch.cuda.memory_reserved()<cfg['GPU_reserved_budget_bytes']
                with torch.no_grad():z=model(images.cuda()).cpu().numpy()
                xs.append(z);ys.append(labels.numpy());encodes+=len(z)
                write(out/'precision_status.private.json',dict(phase='encoding',cudnn_tf32=setting,diagnostic_validation_encodes=encodes,updated=time.time()))
            x=np.concatenate(xs);y=np.concatenate(ys);s=x@w;pred=np.array(seen)[s.argmax(1)];features.append(x);predictions.append(pred);score_sets.append(s)
            records.append(dict(cudnn_tf32=setting,correct={str(c):int((pred[y==c]==c).sum()) for c in seen},validation_n=len(y)))
        q=np.load(out/'query.private.npz');mask=np.isin(q['labels'],seen);assert np.array_equal(q['labels'][mask],y)
        historical=json.loads(Path(cfg['historical_metrics']).read_text());expected={k:round(v*15) for k,v in historical['stages'][0]['per_class_recall'].items()};previous=json.loads((out/'TASK1_PARITY.json').read_text())['actual']
        changed=np.flatnonzero(predictions[0]!=predictions[1]);events=[]
        for i in changed:
            c=int(y[i]);column=seen.index(c)
            events.append(dict(true_class=c,original_setting_prediction=int(predictions[0][i]),F1_setting_prediction=int(predictions[1][i]),original_true_margin=float(score_sets[0][i,column]-np.max(np.delete(score_sets[0][i],column))),F1_true_margin=float(score_sets[1][i,column]-np.max(np.delete(score_sets[1][i],column))),feature_max_delta=float(abs(features[0][i]-features[1][i]).max())))
        confirmed=records[0]['correct']==expected and records[1]['correct']==previous and len(changed)==1 and int(y[changed[0]])==6
        np.savez(out/'precision_evidence.private.npz',original_features=features[0],F1_features=features[1],original_scores=score_sets[0],F1_scores=score_sets[1],labels=y,changed_indices=changed)
        write(out/'PRECISION_AUDIT.json',dict(status='CAUSE_CONFIRMED' if confirmed else 'UNRESOLVED',startup_flags=flags,records=records,original_expected_counts=expected,F1_recorded_counts=previous,original_counts_exact=records[0]['correct']==expected,F1_counts_exact=records[1]['correct']==previous,changed_prediction_count=len(changed),changed_events=events,feature_global_max_delta=float(abs(features[0]-features[1]).max()),F1_prefix_features_match_saved_exact=np.array_equal(features[1],q['x'][mask]),F1_prefix_features_vs_saved_max_delta=float(abs(features[1]-q['x'][mask]).max()),diagnostic_validation_encodes=encodes,model_updates=0,head_fits=0,F1_scientific_acceptance_unchanged=False,worker_seconds=time.time()-start,GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved()))
    finally:
        torch.backends.cudnn.allow_tf32=flags['cudnn_tf32']
        write(out/'PRECISION_COSTS.json',dict(diagnostic_validation_encodes=encodes,new_training_encodes=0,new_optimizer_updates=0,new_head_fits=0,worker_seconds=time.time()-start))
if __name__=='__main__':main()
