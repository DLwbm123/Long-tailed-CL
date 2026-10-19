"""Reader binding for future jobs; does not modify historical extractors or installs."""
EXPECTED={'Pillow':'10.4.0','jpg_abi':'6.2','libjpeg_turbo':'3.0.3'}
def signature():
 from PIL import Image,features
 return dict(Pillow=Image.__version__,jpg_abi=features.version('jpg'),libjpeg_turbo=features.version('libjpeg_turbo'))
def require_canonical_reader():
 actual=signature()
 if actual!=EXPECTED:raise RuntimeError('DECODER_BINDING_MISMATCH: use qualified original reader before image access')
 return actual
if __name__=='__main__':
 import json
 actual=signature()
 try:require_canonical_reader();accepted=True
 except RuntimeError:accepted=False
 print(json.dumps(dict(actual=actual,expected=EXPECTED,accepted=accepted)))
