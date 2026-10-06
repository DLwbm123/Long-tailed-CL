"""Training-calibrated margin guard over three fixed coherent drift blends."""
import numpy as np
from prototype_graph import transport, move, append
from prototype_analytic import ridge

ALPHAS = (0., .5, 1.)


def heads(bank, before, after, y, classes):
    graph, _ = transport(bank, before, after, y, classes)
    global_shift = np.mean([np.mean(np.asarray(after[y==c],dtype=np.float64)-before[y==c],axis=0) for c in classes],axis=0)
    graph_shifts = np.array([a['center']-b['center'] for a,b in zip(graph['components'],bank['components'])])
    result = []
    for alpha in ALPHAS:
        shifted = move(bank, global_shift+alpha*(graph_shifts-global_shift))
        candidate = append(shifted, after, y, classes)
        W, residual = ridge(candidate)
        result.append(dict(alpha=alpha,head=W.astype(np.float32),bank=candidate,ridge_residual=residual))
    return result


def class_margin_losses(scores, labels, evaluated_classes, competitors):
    """Mean squared nonpositive-margin penalty, separately for each protected class."""
    losses = []
    for c in evaluated_classes:
        selected = labels == c
        if not selected.any():raise ValueError('Protected class lacks calibration observations')
        rival = [j for j in competitors if j != c]
        if not rival:raise ValueError('No competing class')
        margin = scores[selected,c]-scores[selected][:,rival].max(1)
        losses.append(float(np.mean(np.maximum(-margin,0.)**2)))
    return np.array(losses)


def select_guard(current_losses, old_losses):
    current_losses,old_losses=np.asarray(current_losses),np.asarray(old_losses)
    if current_losses.shape[0]!=3 or old_losses.shape[0]!=3 or not np.isfinite(current_losses).all() or not np.isfinite(old_losses).all():
        raise ValueError('Invalid fixed-candidate losses')
    eligible = ((current_losses<=current_losses[0]+1e-8).all(1) & (old_losses<=old_losses[0]+1e-8).all(1))
    return int(np.flatnonzero(eligible)[-1]), eligible


if __name__=='__main__':
    for current,old,expected in [([[1,1],[.9,.9],[1.1,.8]],[[1],[.9],[.8]],1),
        ([[1,1],[.9,.9],[.8,.8]],[[1],[1.1],[1.2]],0),
        ([[1,1],[.9,.9],[.8,.8]],[[1],[.9],[.8]],2)]:
        index,_=select_guard(current,old);assert index==expected
    scores=np.array([[1.,0.,2.],[0.,1.,3.]])
    assert np.array_equal(class_margin_losses(scores,np.array([0,1]),[0,1],[2]),[1.,4.])
    from prototype_graph import empty
    rng=np.random.default_rng(63);x=rng.normal(size=(64,5));y=np.repeat([0,1],32)
    bank=append(empty(5),x,y,[0,1]);z=rng.normal(size=(64,5));new_y=np.repeat([2,3],32)
    options=heads(bank,z,z+.2,new_y,[2,3])
    assert all(np.allclose(options[0]['bank']['S'],o['bank']['S']) for o in options)
    fixed,_=transport(bank,z,z+.2,new_y,[2,3]);expected=append(fixed,z+.2,new_y,[2,3])
    assert np.allclose(options[-1]['bank']['S'],expected['S'])
    print('PASS: per-class current/old guard, fallback, strongest admissible blend, coherent uniform moments')
