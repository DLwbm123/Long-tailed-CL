"""Independent NumPy native attention/MLP, reused explicit equations; no old artifacts."""
import numpy as np

def np_tokens(tokens,state,prompts=None,heads=12,eps=1e-6,apply_norm=True):
    """Independent NumPy attention, exact GELU and deep-prompt routing."""
    from scipy.special import erf
    s={k:np.asarray(v,dtype=np.float64) for k,v in state.items()};x=np.asarray(tokens,dtype=np.float64)
    def norm(z,p):
        d=z-z.mean(-1,keepdims=True)
        return d/np.sqrt((d*d).mean(-1,keepdims=True)+eps)*s[p+'weight']+s[p+'bias']
    depth=len([k for k in s if k.startswith('blocks.') and k.endswith('.attn.qkv.weight')])
    for j in range(depth):
        n=0
        if prompts is not None and j<len(prompts):
            n=len(prompts[j]);x=np.concatenate([x[:,:1],np.broadcast_to(np.asarray(prompts[j],dtype=np.float64),(len(x),n,x.shape[-1])),x[:,1:]],1)
        p=f'blocks.{j}.';z=norm(x,p+'norm1.');qkv=z@s[p+'attn.qkv.weight'].T+s[p+'attn.qkv.bias'];b,t,triple=qkv.shape;d=triple//(3*heads)
        q,k,v=qkv.reshape(b,t,3,heads,d).transpose(2,0,3,1,4);a=(q@k.swapaxes(-1,-2))/np.sqrt(d);a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        z=(a@v).transpose(0,2,1,3).reshape(b,t,heads*d);x=x+z@s[p+'attn.proj.weight'].T+s[p+'attn.proj.bias']
        z=norm(x,p+'norm2.');z=z@s[p+'mlp.fc1.weight'].T+s[p+'mlp.fc1.bias'];z=.5*z*(1+erf(z/np.sqrt(2)));x=x+z@s[p+'mlp.fc2.weight'].T+s[p+'mlp.fc2.bias']
        if n:x=np.concatenate([x[:,:1],x[:,1+n:]],1)
    return norm(x,'norm.') if apply_norm else x
