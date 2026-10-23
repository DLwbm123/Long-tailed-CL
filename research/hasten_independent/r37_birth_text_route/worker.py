"""Class-birth text adapter routing; original image route/losses/replay."""
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
for label in ['text','leaf']:
 needle=f'net.Text_encode({label})';assert source.count(needle)==1
 source=source.replace(needle,f'net.Text_encode({label},torch.arange(n,device={label}.device))')
needle='    def start_audit(self):';assert source.count(needle)==1
reference="""
        self._routed_old_reference=None
        if self._known_classes:
            with torch.no_grad():
                net=self._network;ids=torch.arange(self._known_classes,device='cuda')
                raw=net.model.encode_text(net.tokenizer([LEAF_TEMPLATE.format(v) for v in NAMES[:self._known_classes]]).cuda())
                self._routed_old_reference=net.Text_encode(raw,ids).detach().clone()
"""
source=source.replace(needle,needle+reference)
needle='    def _inspect_loss(self,total,contrast,anchor,hier):';assert source.count(needle)==1
proof="""
        net=self._network;old=self._known_classes;n=self._total_classes
        a=self.audit.setdefault('birth_text_route',dict(calls=0,old_rows=0,new_rows=0,old_euclidean_max_change=0.,explicit_class_ids=True,extra_parameters=0))
        a['calls']+=1;a['old_rows']+=old;a['new_rows']+=n-old
        with torch.no_grad():
            raw=net.model.encode_text(net.tokenizer([LEAF_TEMPLATE.format(v) for v in NAMES[:n]]).cuda())
            if old:
                current=net.Text_encode(raw[:old],torch.arange(old,device='cuda'))
                diff=float((current-self._routed_old_reference).abs().max())
                a['old_euclidean_max_change']=max(a['old_euclidean_max_change'],diff);assert diff==0
        if self.audit['steps']==0:
            params=list(net.Text_Adapter[-1].parameters())
            if old:
                probe=raw[:old].detach().requires_grad_(True)
                z=net.Text_encode(probe,torch.arange(old,device='cuda')).square().mean()
                gg=torch.autograd.grad(z,params,allow_unused=True)
                assert all(g is None or torch.count_nonzero(g)==0 for g in gg)
            gg=torch.autograd.grad(total,params,allow_unused=True,retain_graph=True)
            assert all(g is None or torch.isfinite(g).all() for g in gg)
            norm=sum(float(g.detach().double().square().sum()) for g in gg if g is not None)**.5
            assert norm>0
            a['old_prompt_new_adapter_gradient_zero']=True;a['current_text_original_loss_gradient_norm']=norm;a['routing_gradient_proof']=True
"""
source=source.replace(needle,needle+proof)
prior_key='reused_from_r03' if cfg['dataset']=='NIH' else 'reused_prior'
needle=f"if not r.get('{prior_key}',False)"
assert source.count(needle)==1
source=source.replace(needle,"if not r.get('reused_from_r28_initial',False)")
exec(compile(source,'<frozen-dataset-runner>','exec'),dict(__name__='__main__',__file__=cfg['base_runner']))
