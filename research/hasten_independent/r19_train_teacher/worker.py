"""Reuse H0 runners with one additive TRAIN-only teacher anchor in local compat."""
import os,json,sys
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());source=Path(cfg['base_runner']).read_text()
sys.path.insert(0,str(Path(__file__).parent))
os.environ['TEACHER_ROOT']=cfg['teacher_root'];os.environ['TEACHER_DATASET']=cfg['teacher_dataset']
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
exec(compile(source,'<frozen-dataset-runner>','exec'),dict(__name__='__main__',__file__=cfg['base_runner']))
