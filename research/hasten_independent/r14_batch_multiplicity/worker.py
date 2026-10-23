"""Reuse frozen dataset runners; the loss replacement lives only in local compat."""
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
exec(compile(source,'<frozen-dataset-runner>','exec'),dict(__name__='__main__',__file__=cfg['base_runner']))
import compat
(Path(cfg['output'])/'LOSS_MULTIPLICITY_AUDIT.json').write_text(json.dumps(compat.LOSS_AUDIT,indent=2))
