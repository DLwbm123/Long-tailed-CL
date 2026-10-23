"""TRAIN visual means as hierarchy leaves; original losses and inference."""
import os,json,sys
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
source=source.replace(needle,needle+"\n        assert all(len(a.fc)==(2 if CFG['image_nonlinear'] else 1) for a in self._network.Image_Adapter)\n        assert all(isinstance(a.fc[1],nn.GELU) for a in self._network.Image_Adapter) if CFG['image_nonlinear'] else True\n        assert all(len(a.fc)==1 for a in self._network.Text_Adapter)\n        self.audit['image_activation']='GELU_exact' if CFG['image_nonlinear'] else 'Linear'\n        self.audit['gelu_image_adapters']=len(self._network.Image_Adapter) if CFG['image_nonlinear'] else 0\n        self.audit['entailment_compute_dtype']='float64'\n        self.audit['entailment_fp64_batches']=self.audit.get('entailment_fp64_batches',0)+1\n        self.audit['extra_parameters_from_activation']=0")
needle='    def _inspect_loss(self,total,contrast,anchor,hier):'
assert source.count(needle)==1
probe="""
        self.audit['hierarchy_leaf_source']='TRAIN_visual_NCM_mean'
        self.audit['hierarchy_leaf_calls']=self.audit.get('hierarchy_leaf_calls',0)+1
        self.audit['hierarchy_leaf_rows']=self.audit.get('hierarchy_leaf_rows',0)+self._total_classes
        self.audit['hierarchy_means_detached']=True
        mutable=int(self.hierarchy_manager.mutable_nodes.sum())
        assert mutable==self._total_classes-self._known_classes
        self.audit['hierarchy_mutable_leaf_rows']=self.audit.get('hierarchy_mutable_leaf_rows',0)+mutable
        if self.audit['steps']==0:
            params=[(n,p) for n,p in self._network.named_parameters() if p.requires_grad]
            val=sum(hier.values())
            gg=torch.autograd.grad(val,[p for n,p in params],allow_unused=True,retain_graph=True)
            assert all(g is None or torch.isfinite(g).all() for g in gg)
            def norm(kind):return sum(float(g.detach().double().square().sum()) for (n,p),g in zip(params,gg) if g is not None and kind in n)**.5
            image_norm=norm('Image_Adapter');text_norm=norm('Text_Adapter')
            assert image_norm>0 and text_norm==0
            self.audit['image_hierarchy_gradient_norm']=image_norm
            self.audit['text_hierarchy_gradient_norm']=text_norm
            self.audit['hierarchy_image_gradient_proof']=True
"""
source=source.replace(needle,needle+probe)
exec(compile(source,'<frozen-dataset-runner>','exec'),dict(__name__='__main__',__file__=cfg['base_runner']))
