"""A synthetic two-block attention fixture, not a CLIP/native or efficacy test."""
import itertools
import json
import math
import time


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def unit(x):
    n=math.sqrt(dot(x,x))
    return [v/max(n,1e-12) for v in x]
def linear(x,w):return [dot(row,x) for row in w]
def softmax(x):
    z=[math.exp(v-max(x)) for v in x]
    return [v/sum(z) for v in z]


Q=[[.8,.2,-.1],[-.2,.7,.3],[.1,-.3,.9]]
K=[[.6,-.1,.2],[.3,.8,-.2],[-.2,.1,.7]]
V=[[.9,.1,.2],[-.1,.6,.3],[.2,-.2,.8]]
_calls=0
ATOMS=[None,[1.,-.5,.2],[-.4,.8,.5],[.2,.1,-.9]]


def block(tokens):
    out=[]
    for x in tokens:
        weights=softmax([dot(linear(x,Q),linear(z,K))/math.sqrt(3) for z in tokens])
        value=[sum(w*linear(z,V)[j] for w,z in zip(weights,tokens)) for j in range(3)]
        residual=[a+b for a,b in zip(x,value)]
        out.append([v+.2*math.tanh(v) for v in residual])
    return out


def encode(image,program):
    global _calls
    _calls+=1
    tokens=[[.1,-.2,.3],list(image)]
    for action in program:
        # NOOP really omits the prompt token. An appended zero token is not NOOP.
        if action:tokens.append(ATOMS[action].copy())
        tokens=block(tokens)
    return unit(tokens[0])


def run():
    cpu=time.process_time();wall=time.perf_counter();start_calls=_calls
    support=[[.8,.2,-.1],[-.3,.7,.2],[.1,-.4,.9]]
    query=[.1,.2,.1]
    programs=list(itertools.product(range(4),repeat=2))
    rows=[]
    for program in programs:
        prototypes=[encode(s,program) for s in support]
        q=encode(query,program)
        scores=[dot(q,p) for p in prototypes]
        assert all(math.isfinite(v) for v in scores)
        # A single support's prototype is still exactly its own encoded vector.
        assert prototypes[0]==encode(support[0],program)
        rows.append(scores)
    base=rows[0]
    maximum=max(abs(s-b) for row in rows for s,b in zip(row,base))
    assert maximum>1e-8
    assert encode(query,(0,0))==unit(block(block([[.1,-.2,.3],query]))[0])
    # Slots alter different blocks; equal multisets need not mean equal programs.
    order_effect=max(abs(a-b) for a,b in zip(encode(query,(1,2)),encode(query,(2,1))))
    assert order_effect>1e-8
    return {'passed':True,'scope':'synthetic representation freedom only',
            'programs':16,'support_per_class':1,'classes':3,
            'same_program_for_support_and_query':True,'genuine_noop':True,
            'max_score_change_vs_noop':maximum,'slot_order_effect':order_effect,
            'toy_encoder_calls':_calls-start_calls,'new_real_encoder_classifier_forward':0,
            'optimizer_steps':0,'VAL_TEST_reads':0,
            'accuracy_improvement_claim':False,'native_CLIP_pass':False,
            'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall}


if __name__=='__main__':print(json.dumps(run(),indent=2))
