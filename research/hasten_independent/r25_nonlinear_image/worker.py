"""GELU image adapters; original H0 objectives, replay and readout."""
import os,json,sys
if os.environ.get("R25_TRACE_NONFINITE")=="1":
    import torch
    torch.autograd.set_detect_anomaly(True,check_nan=True)
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());source=Path(cfg['base_runner']).read_text()
sys.path.insert(0,str(Path(__file__).parent))
if cfg['dataset']=='NIH':
 old="ARMS=['H-raw','H-unit','H-unit-no-proj'];SEEDS=[74002,74003,74004]"
 assert source.count(old)==1;source=source.replace(old,"ARMS=['H-unit'];SEEDS=[74002,74003,74004]")
else:
 start=source.index('    def ensure(self,stage):');end=source.index('    def release(self):',start)
 source=source[:start]+'''    def ensure(self,stage):
        target=Path(CFG['shared_cache'])/f'stage_{stage}.private.pt'
        rec=torch.load(target,weights_only=False,map_location='cpu')
        for k in ['x','text']:
            assert torch.allclose(rec[k].norm(dim=-1),torch.ones(len(rec[k])),atol=2e-6)
        return rec
'''+source[end:]
# Validate only the cache for the currently requested stage, never open future features.
needle="        rec=torch.load(target,weights_only=False,map_location='cpu')"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n        expected=[i for i,r in enumerate(self.rows) if r['split']=='train' and r['label'] in STAGES[stage]]\n        assert rec['ids']==expected, 'TRAIN_CACHE_ROW_MISMATCH'\n        write(f'cache_stage_{stage}_acceptance.json',dict(rows=len(expected),stage=stage,exact_ids=True,validation_rows=0))")
# Migration uses existing same-encoder unit VAL features only after all TRAIN seals.
start=source.index('    @torch.no_grad()\n    def encode(self,ids,phase):')
end=source.index('    def ensure(self,stage):',start)
source=source[:start]+'''    @torch.no_grad()
    def encode(self,ids,phase):
        assert phase=='ENCODE_VAL_AFTER_ALL_HEADS_SEALED'
        seal=json.loads((OUT/'ALL_HEADS_SEALED.json').read_text())
        assert seal['stages']==3*len(STAGES) and seal['validation_rows_read']==0
        rec=torch.load(CFG['validation_cache'],map_location='cpu',weights_only=False)
        assert rec['ids']==ids and rec['x'].shape==(len(ids),512)
        finite(rec['x']);assert torch.allclose(rec['x'].norm(dim=1),torch.ones(len(ids)),atol=2e-6)
        write('MIGRATION_VAL_CACHE_RECEIPT.json',dict(rows=len(ids),exact_ids=True,unit_norm=True,from_fixed_R16_same_encoder=True,read_after_all_heads_sealed=True,no_reencoding=True,no_test=True))
        return rec['x']
'''+source[end:]
# R16 cache stores exactly the unit features produced by the original eval normalization.
old="eval_x=torch.nn.functional.normalize(x[ids],dim=-1) if encoder.unit else x[ids]"
assert source.count(old)==1;source=source.replace(old,'eval_x=x[ids]')
needle='    def _before_step(self):'
assert source.count(needle)==1
source=source.replace(needle,needle+"\n        assert all(len(a.fc)==2 and isinstance(a.fc[1],nn.GELU) for a in self._network.Image_Adapter)\n        assert all(len(a.fc)==1 for a in self._network.Text_Adapter)\n        self.audit['image_activation']='GELU_exact'\n        self.audit['gelu_image_adapters']=len(self._network.Image_Adapter)\n        self.audit['extra_parameters_from_activation']=0")
if os.environ.get('R25_TRACE_ACTUAL_ENTAIL')=='1':
    import compat
    original_load=compat.load_upstream
    def instrumented_load(root):
        inc,up,geo=original_load(root)
        up._trace_entail_calls=[]
        up._original_entail=up.compute_stable_entailment_loss
        def captured(parent,child,curv,*args,**kw):
            up._trace_entail_calls.append((parent.detach().clone(),child.detach().clone(),curv.detach().clone() if torch.is_tensor(curv) else curv,args,kw))
            return up._original_entail(parent,child,curv,*args,**kw)
        up.compute_stable_entailment_loss=captured
        return inc,up,geo
    compat.load_upstream=instrumented_load
    source=source.replace('    def _after_step(self):','    def _after_step(self):\n        UP._trace_entail_calls.clear()')
if os.environ.get('R25_TRACE_COMPONENTS')=='1':
    import torch
    original_backward=torch.Tensor.backward
    def retained_backward(self,*args,**kwargs):
        kwargs['retain_graph']=True
        return original_backward(self,*args,**kwargs)
    torch.Tensor.backward=retained_backward
    needle='    def _inspect_loss(self,total,contrast,anchor,hier):'
    assert source.count(needle)==1
    source=source.replace(needle,needle+"\n        self._trace_losses=dict(total=total,contrast=contrast,anchor=anchor,**{'hierarchy_'+k:v for k,v in hier.items()})")
    needle='    def _before_step(self):'
    trace="""
        bad=[n for n,p in self._network.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            params=[(n,p) for n,p in self._network.named_parameters() if p.requires_grad]
            components={}
            for k,val in self._trace_losses.items():
                if not val.requires_grad:continue
                gg=torch.autograd.grad(val,[p for n,p in params],allow_unused=True,retain_graph=True)
                invalid=[n for (n,p),g in zip(params,gg) if g is not None and not torch.isfinite(g).all()]
                components[k]=dict(loss=float(val.detach()) if torch.isfinite(val) else None,nonfinite_parameter_names=invalid)
            if os.environ.get('R25_TRACE_ACTUAL_ENTAIL')=='1':
                probes=[]
                for parent,child,curv,args,kw in UP._trace_entail_calls:
                    cases=[]
                    for promote in [False,True]:
                        pp=(parent.double() if promote else parent).detach().requires_grad_(True)
                        cc=(child.double() if promote else child).detach().requires_grad_(True)
                        cv=curv.double() if promote and torch.is_tensor(curv) else curv
                        val=UP._original_entail(pp,cc,cv,*args,**kw)
                        gg=torch.autograd.grad(val,[pp,cc],allow_unused=True)
                        cases.append(dict(promoted_fp64=promote,input_dtype=str(pp.dtype),loss=float(val) if torch.isfinite(val) else None,finite_gradients=all(g is None or bool(torch.isfinite(g).all()) for g in gg)))
                    probes.append(dict(rows=len(parent),parent_norm_min=float(parent.norm(dim=-1).min()),parent_norm_max=float(parent.norm(dim=-1).max()),cases=cases))
                write('TRACE_ACTUAL_ENTAIL.private.json',dict(completed_steps=self.audit['steps'],calls=probes,no_optimizer_changes=True,actual_failed_batch=True))
            write('TRACE_GRADIENT_COMPONENTS.private.json',dict(seed=self.seed,stage=self._cur_task,completed_steps=self.audit['steps'],nonfinite_parameter_names=bad,components=components,mathematics_unchanged=True))
"""
    source=source.replace(needle,needle+trace)
exec(compile(source,'<frozen-dataset-runner>','exec'),dict(__name__='__main__',__file__=cfg['base_runner']))
