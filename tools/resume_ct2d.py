"""One bounded recovery after a container restart; retain the original CT2-D worker."""
import gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_ct2d as base

PARTIAL=('ISIC',1994,'U')
PENDING=[('ISIC',1994,'R'),('ISIC',1995,'U'),('ISIC',1995,'R')]

def verify_preserved(root,lock):
    for name,digest in lock['preserved_score_sha256'].items():
        assert base.sha(root/'output/private/scores'/name)==digest,('BLOCKED_PRESERVED_SCORE_DRIFT',name)
    for name,digest in lock['cache_sha256'].items():
        assert base.sha(root/'output/private/tmp'/name)==digest,('BLOCKED_RECOVERY_CACHE_DRIFT',name)
    assert base.sha(root/'output/private/parent.pt')==lock['parent_sha256']

def cache(r,p,split):
    ds=r.dataset('ISIC',1994,split)
    raw=np.load(r.private/'tmp'/('post_'+split+'.npy'),mmap_mode='r')
    assert raw.shape==(len(ds),2,768) and raw.dtype==np.float32 and np.isfinite(raw).all()
    return dict(raw=raw,y=np.array([x['target'] for x in ds.rows]),rows=ds.rows)

class Recovery(base.Diagnosis):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.recovery=base.read(self.pub/'RECOVERY_LOCK_R1.json')
        assert base.sha(Path(__file__))==cfg['recovery_code_sha256']
        verify_preserved(self.root,self.recovery)
        self.prior=self.recovery['last_ledger']
        self.calls=dict(self.prior['calls'])
        # Four returned ridge solves are evidenced by the retained Q files.
        self.count('new_analytic_fits',4)
        self.calls['CPU_analytic_milliseconds']+=int(1000*self.recovery['unobserved_interval_upper_seconds'])
        self.prior=dict(self.prior,GPU_process_residence_seconds=self.recovery['prior_GPU_upper_seconds'])
        self.gpu_started=time.monotonic()  # Account the entire recovery worker, including transfer/CPU work.
    def ledger(self,enforce=True):
        x=super().ledger(enforce)
        x.update(recovery_attempt=1,resource_values_are_conservative_upper_bounds=True,
            GPU_residence_unobserved_upper_seconds=self.recovery['prior_GPU_upper_seconds']-self.recovery['last_ledger']['GPU_process_residence_seconds'],
            CPU_analytic_unobserved_upper_seconds=self.recovery['unobserved_interval_upper_seconds'],
            analytic_fit_count_lower=x['new_analytic_fits'],analytic_fit_count_upper=x['new_analytic_fits']+2,
            analytic_fit_count_note='Completed solves counted; at interruption an independent solve and Risk solve may have completed without durable records (0..2 additional).',
            peak_GPU_allocated_bytes=max(x['peak_GPU_allocated_bytes'],self.recovery['last_ledger']['peak_GPU_allocated_bytes']))
        base.write(self.pub/'RESOURCE_AND_ACCESS_LEDGER.json',x);return x
    def save_score(self,p,mode,W,val):
        target=self.private/'scores'/f'{p["dataset"]}_{p["seed"]}_{p["stream"]}_{mode}.npz'
        if not target.exists():return super().save_score(p,mode,W,val)
        assert (p['dataset'],p['seed'],p['stream'])==PARTIAL and mode in ('Q00','Q10','Q01','Q11')
        with np.load(target) as f:
            assert list(f['ids'])==[x['sample_id'] for x in val['rows']]
            np.testing.assert_array_equal(f['order'],p['order']);np.testing.assert_array_equal(f['y'],val['y'])
            np.testing.assert_allclose(W,f['W'],atol=1e-12,rtol=1e-12)
            out=base.joint(val['raw'])@W
            np.testing.assert_allclose(out,f['raw'],atol=1e-12,rtol=1e-12)
            return f['raw'].copy()  # Preserve pre-interruption bytes; never replace with the rerun.

def finish_model(r,p,tr,va,net=None):
    assert p['seed']!=1993  # All four transition and eight gradient states already finished.
    began=time.monotonic();base.oracle(r,net,p,tr,va)
    r.count('CPU_analytic_milliseconds',int(1000*(time.monotonic()-began)))
    if p['stream']=='R':base.synthetic_diagnostic(r,net,p,tr)
    if net is not None:assert base.network_hash(net)==p['network_sha256']
    r.ledger()

def run(cfg):
    r=Recovery(cfg)
    assert not (r.pub/'RECOVERY_STARTED_R1.json').exists(),'BLOCKED_RECOVERY_ALREADY_STARTED'
    base.write(r.pub/'RECOVERY_STARTED_R1.json',dict(status='RUNNING',unix=time.time(),source_commit=cfg['recovery_source_commit']))
    try:
        p=torch.load(r.private/'parent.pt',map_location='cpu',weights_only=False)
        assert (p['dataset'],p['seed'],p['stream'],p['task'])==(*PARTIAL,3)
        tr=cache(r,p,'train');va=cache(r,p,'val')
        finish_model(r,p,tr,va)
        base.write(r.pub/'RECOVERED_CACHE_VALIDATION.json',dict(status='PASS',models=1,
            original_parent_SHA=True,locked_manifest_SHA=True,existing_four_weights_and_scores_reproduced=True,
            current_moments_and_Q00_original_reproduced=True,additional_image_reads=0,
            note='Cache hashes locked retrospectively; provenance also checked against original Q00/Q10/Q01/Q11 weights, scores, parent and chronological extraction records.'))
        del tr,va,p;gc.collect()
        for f in (r.private/'tmp').glob('*.npy'):f.unlink()
        (r.private/'parent.pt').unlink();print('MODEL_COMPLETE ISIC 1994 U (retained cache)',flush=True)
        for name,seed,stream in PENDING:
            p,e=r.acquire(name,seed,stream,4);net,_=r.restore(p,e);r.pointwise(net)
            tr=r.extract(net,p,'train','post');va=r.extract(net,p,'val','post')
            finish_model(r,p,tr,va,net)
            del net,tr,va,p;gc.collect();torch.cuda.empty_cache()
            for f in (r.private/'tmp').glob('*.npy'):f.unlink()
            (r.private/'parent.pt').unlink();r.ledger();print('MODEL_COMPLETE',name,seed,stream,flush=True)
        for name,digest in r.recovery['preserved_score_sha256'].items():assert base.sha(r.private/'scores'/name)==digest
        assert len(base.lines(r.pub/'reproduction_audit.jsonl'))==12
        assert len(base.lines(r.pub/'gradient_probes.jsonl'))==8
        assert len(list((r.private/'scores').glob('*_Q*.npz')))==54
        base.write(r.pub/'SCORES_LOCK.json',dict(source_commit=cfg['source_commit'],recovery_source_commit=cfg['recovery_source_commit'],
            files={p.name:base.sha(p) for p in sorted((r.private/'scores').glob('*.npz'))},formal_metric_units=54))
        base.write(r.pub/'GPU_PHASE_COMPLETE.json',dict(status='PASS',models=12,transitions=4,new_training_epochs=0,optimizer_steps=0,recovery_attempt=1))
    except BaseException as e:
        base.write(r.pub/'FAILURE_recovery_R1.json',dict(status='BLOCKED',exception=type(e).__name__,reason=str(e)));raise
    finally:
        r.ledger(False);base.write(r.private/'STOP_TRANSFER.json',dict(status='STOP'))

def report(cfg):
    import report_ct2d
    report_ct2d.main()
    pub=Path(cfg['root'])/'output/public';x=base.read(pub/'RESOURCE_AND_ACCESS_LEDGER.json')
    assert x['GPU_process_residence_seconds']<10800 and x['CPU_analytic_seconds']+600+x['CPU_report_seconds']<7200
    p=pub/'FINAL_REPORT_ZH.md';s=p.read_text()
    s=s.replace(f"新增解析拟合/校验解：{x['new_analytic_fits']}。",f"可证实的解析拟合/校验解：{x['analytic_fit_count_lower']}；中断时另有0至2次未落盘求解，实际总数范围为{x['analytic_fit_count_lower']}–{x['analytic_fit_count_upper']}。")
    s=s.replace('GPU累计驻留：','GPU累计驻留保守上界（含中断未知区间）：')
    s+='\n恢复审计：容器重启，中断前8个模型完整、第9个模型保留4份分数及完整特征。恢复未重读该模型图像，四个已保存解析解逐项核对且未覆盖。仅补余下固定单元；预算使用容器重启时间构造保守上界，无法恢复中断瞬间的精确运行时间及最多2次求解计数。详见 RECOVERY_LOCK_R1.json。\n'
    p.write_text(s)

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    cfg=base.read(os.environ['P21_CONFIG'])
    with threadpool_limits(limits=4):
        if os.environ.get('P21_PHASE')=='report':report(cfg)
        else:run(cfg)

if __name__=='__main__':main()
