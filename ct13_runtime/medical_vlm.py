"""Native medical encoders loaded exclusively from locked local files."""
import copy
import json
from pathlib import Path
import torch
from ct13_runtime.factories import _device, _tensor_digest
from tools.run_nb2_vlm_r1 import _sha

def build_medical(root: Path):
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS
    from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
    root=Path(root)
    lock=json.loads((root/'ASSET_LOCK.json').read_text())
    for name,info in lock['files'].items():
        if _sha(root/name)!=info['sha256']: raise ValueError('MEDICAL_ASSET_CHANGED:'+name)
    source=json.loads((root/'open_clip_config.json').read_text())
    cfg=copy.deepcopy(source['model_cfg'])
    tc=cfg['text_cfg']
    is_bert='hf_model_name' in tc
    if is_bert:
        tc['hf_model_name']=str(root/'text_config')
        tc['hf_tokenizer_name']=str(root)
    name='locked_medical_local'
    _MODEL_CONFIGS[name]=cfg
    device=_device()
    model,_,preprocess=open_clip.create_model_and_transforms(
        name,pretrained=None,pretrained_hf=False,device=device,
        **{'image_'+k:v for k,v in source['preprocess_cfg'].items()})
    if (root/'open_clip_model.safetensors').exists():
        from safetensors.torch import load_file
        state=load_file(str(root/'open_clip_model.safetensors'))
    else:
        state=torch.load(root/'open_clip_pytorch_model.bin',map_location='cpu',weights_only=True)
    position_key='text.transformer.embeddings.position_ids'
    if position_key in state and position_key not in model.state_dict():
        # The official OpenCLIP loader removes this obsolete Transformers buffer.
        # Verify its deterministic value before applying that compatibility rule.
        expected=model.text.transformer.embeddings.position_ids.cpu()
        if not torch.equal(state.pop(position_key),expected):raise ValueError('POSITION_IDS_MISMATCH')
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    if is_bert:
        tokenizer=HFTokenizer(str(root),context_length=tc['context_length'])
    else:
        # Use this model's published tokenizer vocabulary and merge ranks.
        raw=json.loads((root/'tokenizer.json').read_text())['model']
        import gzip
        bpe=root/'native_bpe.txt.gz'
        # Prepared before the protocol is locked and included in ASSET_LOCK.
        if not bpe.exists(): raise ValueError('MISSING_NATIVE_BPE')
        tokenizer=SimpleTokenizer(bpe_path=str(bpe),context_length=tc.get('context_length',77))
        alias={'<|startoftext|>':'<start_of_text>','<|endoftext|>':'<end_of_text>'}
        if any(tokenizer.encoder.get(alias.get(k,k))!=v for k,v in raw['vocab'].items()):
            raise ValueError('MEDICAL_TOKENIZER_VOCAB_MISMATCH')
        if len(tokenizer.encoder)!=len(raw['vocab']):raise ValueError('MEDICAL_TOKENIZER_SIZE')
    initial=_tensor_digest(model.state_dict().items())
    @torch.no_grad()
    def forward(batch):return model.encode_image(batch.to(device)).float().cpu()
    @torch.no_grad()
    def encode_text(tokens):return model.encode_text(tokens.to(device)).float().cpu()
    return {'preprocess':preprocess,'forward':forward,'encode_text':encode_text,'tokenizer':tokenizer,
            'initial_state_hash':initial,'state_hash':lambda:_tensor_digest(model.state_dict().items())}
