"""Deterministic TRAIN metadata component split; no feature/VAL selection."""
def assign_folds(labels,components):
    groups={}
    for i,(y,g) in enumerate(zip(labels,components)):
        if g not in groups:groups[g]=[int(y),[]]
        assert groups[g][0]==int(y),'MULTILABEL_COMPONENT_REQUIRES_EXPLICIT_POLICY'
        groups[g][1].append(i)
    result=[-1]*len(labels)
    for y in sorted(set(map(int,labels))):
        selected=[(g,rows) for g,(c,rows) in groups.items() if c==y]
        assert len(selected)>=2,'INSUFFICIENT_DISTINCT_SUPPORT_QUERY_COMPONENTS'
        sizes=[0,0]
        for g,rows in sorted(selected,key=lambda z:(-len(z[1]),z[0])):
            fold=0 if sizes[0]<=sizes[1] else 1
            for i in rows:result[i]=fold
            sizes[fold]+=len(rows)
        assert min(sizes)>0
    assert all(x in [0,1] for x in result)
    return result
