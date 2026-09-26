"""Resource-only admission qualification before any formal learning."""
import gc,json,os,sys,time,unittest
from pathlib import Path
import numpy as np
import torch
from tools.run_rfvila import Budget,guard,extract,check_models,ROOT
from tools.run_nb2_vlm_r1 import _read,_write,_sha,_manifest,_label,_task_order,_parent_lock,_allow,SIZES
from ct13_runtime.factories import build_apart
from tools.run_medvlm import model_for
from tools.rfvila_reference import projections
from tools.rfvila_math import tensor,solve

def main():
    cfg=_read(Path(os.environ['RF_CONFIG']));out=Path(cfg['run_root']);budget=Budget(out);budget.check('tier')
    assert _read(out/'BACKUP_PREFLIGHT.json')['status']=='PASS' and _read(out/'RESTORE_TEST.json')['status']=='PASS'
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_rfvila_*.py');result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise RuntimeError('PRODUCTION_TESTS_FAILED')
    _write(out/'PRODUCTION_TESTS.json',{'status':'PASS','tests':result.testsRun,'reference_tests':32,'production_test_devices':['cpu','cuda'],'checks':['map parity','class-balanced solve','K lambda','CV nine-grid spectral/reference parity','resume state parity','three-class final task','duplicate class rejection','CSE ties/alpha','metrics']})
    timing={};rng=np.random.default_rng(781)
    X=rng.normal(size=(128,4096))/64;S=X.T@X/len(X);Q=X[:2].T
    for device in ['cpu','cuda']:
        ss=tensor(S,device);qq=tensor(Q,device)
        if device=='cuda':torch.cuda.synchronize()
        start=time.monotonic();w,res=solve(ss,qq,.001)
        if device=='cuda':torch.cuda.synchronize()
        timing[device+'_solve']=time.monotonic()-start
        if device=='cuda':
            start=time.monotonic();torch.linalg.eigh(ss)
            torch.cuda.synchronize();timing['cuda_eigh4096']=time.monotonic()-start
        del ss,qq,w
    maps=out/'private/maps';maps.mkdir(parents=True,exist_ok=True);hashes={}
    for seed in [67101,67102]:
        for name,x in zip(['a','u'],projections(1536,512,4096,seed)):
            p=maps/f'R_{seed}_{name}.npy';np.save(p,x);hashes[p.name]=_sha(p)
    _write(out/'RANDOM_MAP_LOCK.json',{'width':4096,'seeds':[67101,67102],'dtype':'float64','files':hashes,'paired_across_G_B_dataset_parent':True})
    allowed=guard(cfg,'qualify');lock=_read(Path(cfg['task_lock']));models={tag:model_for(tag,cfg) for tag in ['G','B']};receipts=[];total=0;negative=[]
    for ds in SIZES:
        data=cfg['datasets'][ds];root=Path(data['image_root']);train=_manifest(Path(data['train_manifest']),'train',root);total+=len(train)*3
        for seed in [1993,1994,1995]:
            budget.check('tier');order=_task_order(lock,ds,seed);parent,pl=_parent_lock(cfg,lock,ds,seed);apart=build_apart(checkpoint=parent,lock=pl)
            n=48 if seed==1993 else 2
            rows=[r for cid in order[:2] for r in [v for v in train if _label(v)==cid][:n]];_allow(allowed,root,rows)
            # Actual current-task qualification reads are accounted independently of formal fit.
            torch.cuda.synchronize();start=time.monotonic();raw=extract(rows,root,apart,models,out,budget,'qualification',ds,seed,1);torch.cuda.synchronize();elapsed=time.monotonic()-start
            receipts.append({'dataset':ds,'seed':seed,'n':len(rows),'seconds':elapsed,'parent_sha256':pl['parent_sha256'],'feature_dims':[1536,512,512]})
            for path in [Path(data['val_manifest']),root/next(r['relative_path'] for r in train if _label(r) not in order[:2]),out/'private/val.npz',out/'test.npz',out/'reserved.npz']:
                try:
                    with path.open('rb'):pass
                except PermissionError:negative.append(True)
                else:raise AssertionError('ACCESS_NEGATIVE_TEST_FAILED')
            oldpath=root/rows[0]['relative_path'];_allow(allowed,root,[next(r for r in train if _label(r) not in order[:2])])
            try:
                with oldpath.open('rb'):pass
            except PermissionError:negative.append(True)
            else:raise AssertionError('OLD_IMAGE_ACCESS_ALLOWED')
            check_models(apart,models);del apart,raw;gc.collect();torch.cuda.empty_cache()
    check_models(None,models)
    q={'status':'PASS','gpu_name':torch.cuda.get_device_name(0),'physical_gpu':2,'timings':timing,'parents':receipts,'total_formal_train_images':total,'negative_access_tests_passed':len(negative),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'tier_not_selected_from_scores':True,'warmup':'test suite and two model loads before first timed image batch','qualification_image_reads':sum(r['n'] for r in receipts)}
    _write(out/'QUALIFICATION.json',q);print(json.dumps(q),flush=True)

if __name__=='__main__':main()
