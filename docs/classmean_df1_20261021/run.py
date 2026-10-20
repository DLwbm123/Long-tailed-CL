"""Two fixed CPU readouts; no image encoding or regularization search."""
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
import torch


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def statistics(cfg):
    """Read-only input assembly, shared by the two independent solve paths."""
    base = Path(cfg['remote_base'])
    helper = module(base/'joint_heads1_20261019/core.py', 'joint_assembly')
    cls, joint, targets, coverage = [], [], [], []
    support=json.loads((Path(cfg['prior_run'])/'support.json').read_text())
    for c in cfg['order']:
        a = torch.load(Path(cfg['prior_run'])/f'class_{c:02d}.private.pt', weights_only=True)
        b = torch.load(Path(cfg['matched_run'])/f'class_{c:02d}.private.pt', weights_only=True)
        cross = torch.load(Path(cfg['joint_run'])/f'cross_{c:02d}.private.pt', weights_only=True)['cross']
        cls.append(a['ridge_xx'])
        joint.append(helper.joint_gram(a['ridge_xx'], b['patch_mean_gram'], cross))
        targets.append(torch.cat([a['ridge_xy'][:-1], b['patch_mean_target'][:-1], a['ridge_xy'][-1:]]))
        coverage.append(dict(class_id=c, images=support[str(c)]['images'], groups=a['groups']))
        assert torch.allclose(a['ridge_xy'],a['ridge_xx'][:,-1],atol=1e-10,rtol=0)
        assert abs(float(cls[-1][-1,-1])-1)<1e-9
    ec=sum(float(g[:-1,:-1].trace()) for g in cls[:10])/10
    ep=sum(float(g[768:1536,768:1536].trace()) for g in joint[:10])/10
    assert ec>0 and ep>0
    return cls, joint, targets, coverage, (ec/ep)**.5


def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    def status(phase, **kw):
        assert time.time()<cfg['hard_deadline_epoch']-60
        write(root/'status.private.json', dict(state='RUNNING',phase=phase,updated=time.time(),**kw))
    status('math_qualification')
    x=torch.randn(17,7,dtype=torch.float64);y=torch.randn(17,3,dtype=torch.float64)
    g=x.T@x;t=x.T@y;eye=torch.eye(7,dtype=torch.float64)
    for k in [10,20]:
        w=torch.linalg.solve(g+(k/10)*eye,t)
        v=torch.linalg.solve(g/k+.1*eye,t/k)
        assert torch.allclose(w,v,atol=1e-12,rtol=1e-12)
    write(root/'QUALIFICATION.json',dict(state='PASS',mean_risk_equivalence=True,qualification_image_encodes=0,optimizer_updates=0))
    status('load_statistics')
    df=module(Path(cfg['remote_base'])/'joint_controls1_20261019/core.py','df_math')
    cls,joint,targets,coverage,scale=statistics(cfg)
    assert sum(s['images'] for s in coverage)==68058
    audits=[]
    for stage in range(6):
        n=sum(cfg['task_sizes'][:stage+1]);alpha_c=n/10
        cg=sum(cls[:n]);ct=torch.stack([g[:,-1] for g in cls[:n]],1)
        ng,nt,factors=df.scaled(sum(joint[:n]),torch.stack(targets[:n],1),scale)
        target=df.degrees(df.spectrum(cg),alpha_c)
        alpha_j=df.match_alpha(df.spectrum(ng),target)
        heads={'cls_classmean':dict(head=torch.linalg.solve(cg+alpha_c*torch.eye(769,dtype=torch.float64),ct),alpha=alpha_c),
               'joint_classmean_df':dict(head=torch.linalg.solve(ng+alpha_j*torch.eye(1537,dtype=torch.float64),nt),alpha=alpha_j,factors=factors)}
        torch.save(heads,root/f'heads_{stage}.private.pt')
        audits.append(dict(stage=stage,classes=n,CLS_alpha=alpha_c,CLS_mean_risk_penalty=.1,joint_alpha=alpha_j,target_df=target,joint_df=df.degrees(df.spectrum(ng),alpha_j)))
        status('fit_heads',stage=stage)
    write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,heads=12,patch_scale=scale,coverage=coverage,stages=audits))
    # Keep all query access and metrics in the separately launched verifier.
    write(root/'costs.json',dict(new_training_encodes=0,new_validation_encodes=0,qualification_encodes=0,optimizer_updates=0,new_ridge_heads=12,old_heads_refitted=0,attempts=1,wall_seconds=time.time()-cfg['original_start_epoch'],device='CPU'))
    write(root/'status.private.json',dict(state='FIT_COMPLETE_PENDING_VERIFICATION',updated=time.time()))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        root=Path(cfg['output']);root.mkdir(exist_ok=True)
        write(root/'FAILURE.private.json',dict(updated=time.time(),error=traceback.format_exc()))
        raise
