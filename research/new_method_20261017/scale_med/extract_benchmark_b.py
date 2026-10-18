"""Historical 1536-D feature extraction in its original runtime; no fitting."""
import json,os,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F


def main(cfg):
    root=Path(cfg['output']);sys.path.insert(0,str(Path(cfg['legacy_source'])/'tools'))
    from run_multilabel import ApartFeatures
    from run_prototype_single import Images
    from run_medical_v2 import transform,seed_all,write_json
    from torch.utils.data import DataLoader
    torch.set_num_threads(4);seed_all(cfg['seed']);start=time.time()
    model=ApartFeatures(cfg['legacy_source'],cfg['weight'],23,'cuda',cfg['seed']).eval().requires_grad_(False)
    def extract(rows,image_root):
        blocks=[]
        for x,y in DataLoader(Images(rows,image_root,transform(False),cfg['seed']),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode():z=model(x.cuda()).cpu().double()
            assert z.shape[1]==1536 and torch.isfinite(z).all()
            blocks.append(F.normalize(z,dim=1))
        return torch.cat(blocks)
    parity={}
    # Only previously opened Qfit is used for engineering parity, before test inference.
    for n in ['ISIC','HK']:
        prior=Path(cfg['prior_rows']);rows=json.loads((prior/(n+'.Qfit.rows.private.json')).read_text())
        old=torch.load(prior/(n+'.Qfit.features.private.pt'),map_location='cpu',weights_only=True)
        ids=[next(i for i,r in enumerate(rows) if r['label']==c) for c in sorted({r['label'] for r in rows})]
        x=extract([rows[i] for i in ids],cfg['datasets'][n]['development_images'])
        err=float((x-old['x'][ids]).abs().max());assert err<cfg['feature_parity_atol'],f'Historical encoder parity failed: {err}'
        parity[n]=dict(samples=len(ids),max_abs_feature_error=err,tolerance=cfg['feature_parity_atol'])
    write_json(root/'B_FEATURE_PARITY.json',parity)
    for n in ['ISIC','HK']:
        rows=json.loads((root/(n+'.test.rows.private.json')).read_text())
        x=extract(rows,cfg['datasets'][n]['images'])
        torch.save(dict(x=x,y=torch.tensor([r['label'] for r in rows]),components=[r['identity_component'] for r in rows]),root/(n+'.test.features.private.pt'))
        write_json(root/'status.json',dict(state='RUNNING',phase='historical_features',dataset=n,images=len(rows)))
    write_json(root/'B_FEATURE_COST.json',dict(wall_seconds=time.time()-start,model_updates=0,parity_images=21,test_images=1857,torch_version=torch.__version__,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('FEATURES_COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        (Path(cfg['output'])/'failure.private.json').write_text(json.dumps(dict(phase='historical_features',error=traceback.format_exc()),indent=2));raise
