"""Standard-library scalar reference: capacity counterexample and independent derivatives.
No data, models, dependencies, training, scores from previous experiments, or GPU.
"""
import math

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def unit(a):return [x/max(norm(a),1e-12) for x in a]

def reference(patches,q,v):
    h=[unit(x) for x in patches];qh=unit(q);logits=[10*dot(qh,x) for x in h]
    w=[math.exp(t-max(logits)) for t in logits];a=[t/sum(w) for t in w]
    value=[dot(v,x) for x in h];avg=dot(a,value);scale=25/math.sqrt(len(q))
    gv=[scale*sum(a[i]*h[i][j] for i in range(len(h))) for j in range(len(q))]
    raw=[scale*10*sum(a[i]*(value[i]-avg)*h[i][j] for i in range(len(h))) for j in range(len(q))]
    gq=[(raw[j]-dot(raw,qh)*qh[j])/norm(q) for j in range(len(q))]
    return scale*avg,gq,gv

def check():
    p=[[1.,0,0,0],[-1.,0,0,0]];r=[[0.,1,0,0],[0.,-1,0,0]]
    assert [sum(x[j] for x in p) for j in range(4)]==[sum(x[j] for x in r) for j in range(4)]
    qs=[[1.,0,0,0],[0.,1,0,0]];vs=qs
    score=[[reference(bag,q,v)[0] for q,v in zip(qs,vs)] for bag in [p,r]]
    assert score[0][0]>score[0][1] and score[1][1]>score[1][0]
    mixed=[[.3,.7,-.2,.6],[-.4,.8,.5,.2],[.9,-.1,.2,-.4]]
    q=[.7,-.2,.4,.3];v=[.2,.8,-.5,.4];val,gq,gv=reference(mixed,q,v);eps=1e-5;errors=[]
    for target,expected in [(q,gq),(v,gv)]:
        for j in range(4):
            plus=target.copy();minus=target.copy();plus[j]+=eps;minus[j]-=eps
            fp=reference(mixed,plus,v)[0] if target is q else reference(mixed,q,plus)[0]
            fm=reference(mixed,minus,v)[0] if target is q else reference(mixed,q,minus)[0]
            errors.append(abs((fp-fm)/(2*eps)-expected[j]))
    assert max(errors)<1e-7
    assert abs(reference(list(reversed(mixed)),q,v)[0]-val)<1e-12
    assert reference(mixed,q,[0.]*4)[0]==0
    return dict(state='PASS',scope='Scalar4d attention reference only; not Torch, real768d, GPU or NIH evidence.',same_mean_distinct_local_bags_scores=score,finite_difference_max_error=max(errors),permutation_invariant=True,zero_local_bypass=True,new_images=0,new_optimizer_updates=0)

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
