"""Base-only, patient-level appearance cohorts and reproducible crop receipts."""
import random
import numpy as np
import torch
from core import crop

def patient_means(rows,features,record):
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],[]).append(i)
    means={};error=0.
    for patient,ix in groups.items():
        v=features[ix,:1536].mean(0)*(2**.5);v=v/v.norm().clamp_min(1e-12);x=features[ix,:1536].numpy().astype(np.float64);ref=x.sum(0)/len(ix)*(2**.5);ref/=max(np.linalg.norm(ref),1e-12);error=max(error,float(abs(v.numpy()-ref).max()));means[patient]=v
    record(dict(check='independent_equal_image_patient_discovery_means',patients=len(groups),error=error,PASS=error<=1e-12));assert error<=1e-12
    return groups,means

def discover(means,seed,record,C=100,K=64):
    names=sorted(means);assert len(names)>=max(C,K);x=torch.stack([means[g] for g in names]);rng=random.Random(seed);anchors=rng.sample(names,C);coordinates=[];words=[];error=0.
    for anchor in anchors:
        a=names.index(anchor);dist=((x-x[a])**2).sum(1);order=sorted(range(len(names)),key=lambda i:(float(dist[i]),names[i]))[:K]
        ref=((x.numpy()-x[a].numpy())**2).sum(1);ind=sorted(range(len(names)),key=lambda i:(float(ref[i]),names[i]))[:K];error=max(error,float(abs(dist.numpy()-ref).max()));same=order==ind;record(dict(check='independent_patient_nearest_cohort',word=len(words),max_distance_error=error,PASS=same));assert same
        # An anchor has exact distance zero; exact duplicate appearances can tie, so explicitly retain it.
        cohort=[anchor]+[names[i] for i in order if names[i]!=anchor];cohort=cohort[:K]
        while True:
            coord=(rng.randint(8,104),rng.randint(8,104))
            if coord not in coordinates:coordinates.append(coord);break
        words.append(dict(anchor=anchor,patients=cohort,top=coord[0],left=coord[1]))
    return words

def word_draw(words,groups,seed,n=16):
    rng=random.Random(seed);receipt=[]
    for _ in range(n):
        label=rng.randrange(len(words));w=words[label];patient=rng.choice(w['patients']);index=rng.choice(groups[patient]);top=w['top']+rng.randint(-4,4);left=w['left']+rng.randint(-4,4);receipt.append(dict(word=label,patient=patient,index=index,top=top,left=left))
    return receipt

def render(ds,receipt):return torch.cat([crop(ds[r['index']][0][None],r['top'],r['left']) for r in receipt])
