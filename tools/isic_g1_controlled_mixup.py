"""G1: CPU-only, current-arrival analytic fits on locked train/val caches.

No encoder imports. All real features retain their cached float32 values, cast
to float64. Private data paths are supplied by an untracked runtime JSON.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import scipy
from scipy.special import betaincinv
from threadpoolctl import threadpool_info, threadpool_limits

from report_locked_holdout_r1 import analyze_unit, csvwrite, read, write

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL, LAMBDA = 1e-10, 1e-9, 1e-3
COUNTERS = dict(neural_training_epochs=0, optimizer_steps=0,
                encoder_forward_calls=0, new_test_predictions=0,
                new_test_feature_reads=0, validation_adaptively_reused=True,
                full_GSR_reproduction=False, further_experiments_started=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)
    return float(np.max(np.abs(a-b)))


def moments(z):
    assert len(z) > 0 and z.ndim == 2, 'EMPTY_CLASS'
    assert np.isfinite(z).all(), 'NONFINITE_FEATURE'
    return dict(n=len(z), s=z.sum(0), T=z.T @ z)


def mixing(z, method, seed, label):
    """One fixed ring, independent uniform stream, no global RNG mutation."""
    assert len(z) and np.isfinite(z).all(), 'EMPTY_OR_NONFINITE_CLASS'
    n = len(z)
    pi = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, label, 0]))).permutation(n)
    u0 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, label, 1]))).random(n)
    eps = np.finfo(np.float64).eps
    u = np.clip(u0, eps, 1-eps)
    alpha = .6 + .4*np.exp(-.005*n) if method == 'B4' else .6
    gamma = betaincinv(alpha, alpha, u)
    assert np.isfinite(gamma).all() and (gamma >= 0).all() and (gamma <= 1).all()
    j = np.roll(pi, -1)
    assert np.array_equal(np.sort(pi), np.arange(n))
    assert np.array_equal(np.sort(j), np.arange(n))
    assert n == 1 or np.all(pi != j)
    if n == 1:
        aug = z.copy()
    else:
        aug = blend(z[pi], z[j], gamma, method in ('B3', 'B4'))
    a = dict(n=n, M=n, original_label=label, mix_seed=seed, method=method,
             alpha=float(alpha), pair_sha256=digest(np.stack([pi, j], 1)),
             uniform_sha256=digest(u), gamma_sha256=digest(gamma),
             uniform_clip_count=int(np.count_nonzero(u != u0)),
             endpoint_coverage=True, self_pairs=int(np.count_nonzero(pi == j)),
             singleton_policy='SINGLETON_COPY' if n == 1 else 'not_used',
             gamma_min=float(gamma.min()), gamma_max=float(gamma.max()),
             gamma_mean=float(gamma.mean()), gamma_q=np.quantile(gamma, [.05,.5,.95]).tolist())
    return aug, a, pi, j, u, gamma


def blend(x, y, gamma, spherical):
    z = gamma[:, None]*x + (1-gamma[:, None])*y
    assert np.isfinite(z).all(), 'NONFINITE_MIXTURE'
    if spherical:
        norm = np.linalg.norm(z, axis=1)
        assert np.all(norm >= 1e-12), 'DEGENERATE_SPHERICAL_MIXTURE'
        z = z / norm[:, None]
    return z


def contribution(real, aug, method):
    if method == 'R0':
        return real['T'], real['s'], real['n']
    if aug is None:
        return real['T']/real['n'], real['s']/real['n'], 1
    count = real['n']+aug['n']
    return (real['T']+aug['T'])/count, (real['s']+aug['s'])/count, 1


class Increment:
    """The learner receives exactly one current class's sufficient statistics."""
    def __init__(self, order, dim, method):
        self.order, self.method = list(order), method
        self.visited, self.cols, self.counts = [], [], []
        self.H = np.zeros((dim, dim), dtype=np.float64)
        self.scale = 0

    def arrive(self, label, real, aug=None):
        assert label == self.order[len(self.visited)], 'FUTURE_OR_REPEATED_CLASS_ACCESS'
        H, h, weight = contribution(real, aug, self.method)
        self.H += H
        self.cols.append(h)
        self.scale += weight
        self.visited.append(label)
        self.counts.append(real['n'])

    def system(self):
        return self.H/self.scale, np.stack(self.cols, 1)/self.scale

    def save(self, path):
        path=Path(path); temp=path.with_name(path.stem+'.part.npz')
        np.savez(temp, H=self.H, cols=np.stack(self.cols, 1), scale=self.scale,
                 order=self.order, visited=self.visited, counts=self.counts, method=self.method)
        temp.replace(path)

    @classmethod
    def restore(cls, path):
        with np.load(path, allow_pickle=False) as a:
            obj = cls(a['order'].tolist(), len(a['H']), str(a['method']))
            obj.H = a['H'].copy(); obj.cols = list(a['cols'].T.copy())
            obj.scale = int(a['scale']); obj.visited = a['visited'].tolist(); obj.counts = a['counts'].tolist()
        return obj


def solve(G, R):
    A = G + LAMBDA*np.eye(len(G))
    W = np.linalg.solve(A, R)
    residual = float(np.linalg.norm(A@W-R)/np.linalg.norm(R))
    assert np.isfinite(W).all() and residual < 1e-10, 'BLOCKED_SOLVER'
    return W, residual


def eigen_diagnostics(e):
    tol = np.finfo(np.float64).eps*len(e)*float(np.max(np.abs(e)))
    assert e.min() >= -tol, 'NON_PSD_BEYOND_DIAGNOSTIC_TOLERANCE'
    clipped = np.maximum(e, 0)
    p = clipped[clipped > 0]/clipped.sum()
    return dict(trace=float(e.sum()), lambda_min=float(e[0]), lambda_max=float(e[-1]),
                condition=float(e[-1]/e[0]) if e[0] > tol else None,
                condition_reason=None if e[0] > tol else 'zero_or_near_zero_at_reported_rank_threshold',
                stable_rank=float(np.sum(e*e)/np.max(np.abs(e))**2),
                spectral_entropy=float(-np.sum(p*np.log(p))),
                numeric_rank=int(np.sum(e > tol)), numeric_rank_threshold=tol,
                entropy_clipped_negative_count=int(np.sum(e < 0)))


def geometry(real_z, aug_z, label, method, mix_seed, span_basis=None):
    row = dict(method=method, mix_seed=mix_seed, original_label=label, n=len(real_z), M=0 if aug_z is None else len(aug_z))
    means = []
    for name, z in [('real', real_z), ('aug', aug_z)]:
        if z is None:
            continue
        mu = z.mean(0); norm = np.linalg.norm(z, axis=1); means.append(mu)
        row.update({name+'_mean_norm':float(np.linalg.norm(mu)), name+'_second_moment_trace':float(np.mean(norm**2)),
                    name+'_centered_cov_trace':float(np.mean(norm**2)-mu@mu),
                    name+'_feature_norm_quantiles':np.quantile(norm,[0,.05,.5,.95,1]).tolist()})
    joint_mu = means[0] if aug_z is None else (means[0]+means[1])/2
    joint_trace = row['real_second_moment_trace'] if aug_z is None else (row['real_second_moment_trace']+row['aug_second_moment_trace'])/2
    row.update(joint_mean_norm=float(np.linalg.norm(joint_mu)), joint_second_moment_trace=joint_trace,
               joint_centered_cov_trace=float(joint_trace-joint_mu@joint_mu),
               joint_mean_shift_norm=float(np.linalg.norm(joint_mu-means[0])))
    if aug_z is not None and span_basis is not None:
        residual = float(np.linalg.norm(aug_z-(aug_z@span_basis.T)@span_basis)/np.linalg.norm(aug_z))
        assert residual < 1e-9, 'BLOCKED_ROW_SPAN'
        row['aug_row_span_relative_residual'] = residual
        row['real_numerical_row_rank'] = len(span_basis)
    return row, np.stack([means[0], joint_mu] if aug_z is None else [means[0], means[1], joint_mu])


def engineering(out, source_path):
    """Required mathematical tests use local RNG and synthetic train-only data."""
    t = time.monotonic(); rng = np.random.default_rng(71)
    z = rng.normal(size=(13, 8)); z[:, 3:] = 0; z /= np.linalg.norm(z, axis=1, keepdims=True)
    initial = np.random.get_state(); checks = {}; mixes = {}
    for method in ('B2','B3','B4'):
        result = mixing(z,method,41001,6); mixes[method] = result
        assert result[1]['M'] == len(z) and result[1]['self_pairs'] == 0
        close(result[0][:,3:], np.zeros((len(z),5)))
        assert np.max(np.abs(np.linalg.norm(result[0],axis=1)-1)) < 1e-10 if method != 'B2' else np.max(np.linalg.norm(result[0],axis=1)) <= 1+2e-6
        close(mixing(z,method,41001,6)[0],result[0])
    assert np.array_equal(mixes['B2'][5],mixes['B3'][5])
    assert np.array_equal(mixes['B3'][4],mixes['B4'][4]) and np.array_equal(mixes['B3'][2],mixes['B4'][2])
    final = np.random.get_state(); assert all(np.array_equal(a,b) for a,b in zip(initial,final))
    checks['ring_beta_norm_span_and_rng'] = 'PASS'
    for method in ('B2','B3','B4'):
        close(mixing(z[:1],method,41001,6)[0],z[:1])
    for bad in (z[:0], np.full((2,8),np.nan), np.full((2,8),np.inf)):
        try: mixing(bad,'B3',41001,6)
        except AssertionError: pass
        else: raise AssertionError('BOUNDARY_NOT_REJECTED')
    try: blend(z[:1],-z[:1],np.array([.5]),True)
    except AssertionError: pass
    else: raise AssertionError('ZERO_DENOM_NOT_REJECTED')
    checks['singleton_empty_nonfinite_antipodal'] = 'PASS'
    # Independent batch objective, current-only arrivals, and saved-statistic resume.
    zs = [z[:5],z[5:]]; stats = [moments(x) for x in zs]
    for method in ('R0','B0','B1','B2','B3','B4'):
        inc = Increment([6,7],8,method); arrays=[]; weights=[]; targets=[]
        for i,(label,x) in enumerate(zip([6,7],zs)):
            aug = None if method in ('R0','B0') else (x.copy() if method=='B1' else mixing(x,method,41001,label)[0])
            inc.arrive(label,stats[i],None if aug is None else moments(aug))
            xx=x if aug is None else np.concatenate([x,aug]); arrays.append(xx)
            targets.extend([i]*len(xx)); weights.extend([1 if method=='R0' else 1/len(xx)]*len(xx))
            if i == 0:
                G0,R0=inc.system(); assert R0.shape[1]==1 and inc.visited==[6]
                inc.save(out/'resume_selfcheck.npz'); inc = Increment.restore(out/'resume_selfcheck.npz')
        X=np.concatenate(arrays); w=np.array(weights,dtype=np.float64); w/=w.sum(); Y=np.eye(2)[targets]
        G,R=inc.system(); close(G,X.T@(w[:,None]*X)); close(R,X.T@(w[:,None]*Y))
        W,_=solve(G,R); close(W,solve(X.T@(w[:,None]*X),X.T@(w[:,None]*Y))[0]); assert abs(w.sum()-1)<1e-14
        if method=='B0': baseline=(G,R,W)
        if method=='B1':
            for a,b in zip(baseline,(G,R,W)): close(a,b)
        try: inc.arrive(6,stats[0])
        except (AssertionError,IndexError): pass
        else: raise AssertionError('FUTURE_ACCESS_NOT_REJECTED')
    (out/'resume_selfcheck.npz').unlink()
    checks['stream_batch_resume_future_and_duplicate'] = 'PASS'
    # Execute only the AST of the historical pure solve_stats; never import its encoder.
    tree=ast.parse(Path(source_path).read_text()); node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='solve_stats')
    ns={'np':np}; exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_solve_stats>','exec'),ns)
    z768=np.pad(z,((0,0),(0,760))); s={c:dict(n=len(x),s=x.sum(0),Q=x.T@x) for c,x in zip([6,7],[z768[:5],z768[5:]])}
    old=ns['solve_stats'](s,[6,7]); inc=Increment([6,7],768,'B0')
    for c,x in zip([6,7],[z768[:5],z768[5:]]): inc.arrive(c,moments(x))
    W,res=solve(*inc.system()); close(W,old[0]); checks['original_solve_stats_AST_compatibility']='PASS'
    p=dict(raw=np.eye(4),y=np.arange(4),original=np.array([4,0,3,7]),order=np.array([4,0,3,7,5,6,2,1]),
           ids=np.array(list('abcd')),component=np.array(list('abcd')),lesion=np.array(list('abcd')))
    m,pc,err=analyze_unit(p,'B0',1993,0)
    assert m['balanced_accuracy']==np.mean([x['recall'] for x in pc])==100 and m['old_macro_recall'] is None
    assert np.argmax(np.ones(4))==0; checks['mapping_empty_groups_errors_BA_tie']='PASS'
    checks['seconds']=time.monotonic()-t
    return checks


class Run:
    def __init__(self, config):
        self.cfg=config; self.out=Path(config['out']); self.out.mkdir(parents=True,exist_ok=True)
        self.public=self.out/'public'; self.private=self.out/'private'
        self.public.mkdir(exist_ok=True); self.private.mkdir(exist_ok=True)
        self.start=time.monotonic(); self.protocol=read(ROOT/'exps/isic_g1_protocol.json')
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY_REQUIRED'
        self.orders={int(k):v for k,v in self.protocol['class_orders'].items()}
        self.real={}; self.aug={}; self.refs={}; self.span={}; self.means={}
        self.metrics=[]; self.per_class=[]; self.errors=[]; self.spectra=[]; self.projections=[]
        self.sampling=[]; self.geometries=[]; self.arrivals=[]; self.solves=0; self.checks={}; self.final={}
        self.peak_bytes=0; self.min_free=shutil.disk_usage(self.out).free
        write(self.public/'PROTOCOL_G1.json',self.protocol)

    def resources(self):
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        free=shutil.disk_usage(self.out).free
        self.peak_bytes=max(self.peak_bytes,size); self.min_free=min(self.min_free,free)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert size<=256*1024**2 and free>=1024**3, 'BLOCKED_DISK_BUDGET'
        assert time.monotonic()-self.start<7200, 'BLOCKED_CPU_TIME_BUDGET'
        assert rss<=4*1024**3, 'BLOCKED_RSS_BUDGET'
        blas=[dict(r,filepath=Path(r['filepath']).name) for r in threadpool_info()]
        assert all(r['num_threads']<=8 for r in blas)
        return dict(wall_seconds=time.monotonic()-self.start,peak_RSS_bytes=rss,
                    observed_max_new_file_bytes=self.peak_bytes,min_observed_free_bytes=self.min_free,
                    actual_formal_solve_calls=self.solves,BLAS=blas)

    def audit(self):
        t=time.monotonic(); cache=Path(self.cfg['cache']); manifests=Path(self.cfg['manifests'])
        lock=read(cache/'CACHE_LOCK.json'); assert lock['encoder']=='A' and lock['model_sha256']=='c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800'
        assert lock['preprocess']['dimension']==768 and lock['preprocess']['feature']=='final layer norm CLS; sample L2'
        self.data={}; report={}
        for split in ('train','val'):
            spec=lock['splits'][split]; paths=[cache/(split+'.npz'),cache/(split+'_associations.json'),manifests/(split+'.csv')]
            pinned={'train':['75d56237dae5a09de9dcd363960e829b0b5deefc7cf3110459c5c94c98d80998','9c992ba26608abb0c7045f334c1d04bddec29eab426868691762e7b34db9a02d','6c9113aaac70a3d0c976177d28d54fbaa3f0138f0fa74715c225014de6ee8abd'],
                    'val':['b812da9181d2a2a0c2e98e0d6233fc771366820d18192194d400c5fd5f7ed864','553cb7c8b48f8fb833fe35f39482098261e97f645f89497a4ef0cf9671cb5511','f485459bb5636ef85bf5c023d3c8c7b988df2e51cdada25967e1888ab7581956']}
            expected=[spec['cache_sha256'],spec['associations_sha256'],spec['manifest_sha256']]
            assert expected==pinned[split], 'BLOCKED_LOCK_CHAIN'
            for path,want in zip(paths,expected): assert sha(path)==want, 'BLOCKED_FEATURE_CACHE_HASH'
            with np.load(paths[0],allow_pickle=False) as f: z=f['z'].copy(); y=f['y'].copy()
            rows=read(paths[1]); ids=[r['sample_id'] for r in rows]
            assert len(z)==len(rows)==spec['n'] and z.shape[1]==768 and z.dtype==np.float32
            assert ids==sorted(ids) and len(set(ids))==len(ids)
            assert np.array_equal(y,[int(r['target']) for r in rows]) and np.array_equal(y,[int(r['original_label']) for r in rows])
            manifest={r['sample_id']:r for r in csv.DictReader(paths[2].open())}
            assert set(manifest)==set(ids)
            for r in rows:
                assert r['split']==split
                for key in ('original_label','lesion_id','identity_component'): assert str(r[key])==str(manifest[r['sample_id']][key]), 'ASSOCIATION_MISMATCH'
                assert r['identity_component'] and r['lesion_id']
            assert np.isfinite(z).all(); norm_error=float(np.max(np.abs(np.linalg.norm(z.astype(np.float64),axis=1)-1)))
            assert norm_error<=2e-6, 'BLOCKED_CACHE_NORM'
            counts=np.bincount(y,minlength=8).tolist()
            if split=='train': assert counts==self.protocol['train_counts']
            self.data[split]=(z,y,rows)
            report[split]=dict(n=len(z),counts=counts,dimension=768,dtype=str(z.dtype),cache_sha256=expected[0],associations_sha256=expected[1],manifest_sha256=expected[2],max_cached_unit_norm_error=norm_error,
                               n_components=len({r['identity_component'] for r in rows}),sample_order_verified=True,manifest_labels_and_groups_verified=True)
        tr=self.data['train'][2]; va=self.data['val'][2]
        for key in ('sample_id','identity_component','lesion_id'):
            assert not ({r[key] for r in tr}&{r[key] for r in va}), 'TRAIN_VAL_LEAKAGE'
        for rows in (tr,va):
            labels={}
            for r in rows:
                key=r['identity_component']; labels.setdefault(key,set()).add(int(r['original_label']))
            assert all(len(x)==1 for x in labels.values()), 'COMPONENT_LABEL_CONFLICT'
        self.cache=cache
        audit=dict(status='PASS',splits=report,cache_lock_sha256=sha(cache/'CACHE_LOCK.json'),cache_lock=lock,
                   actual_imbalance_ratio=max(self.protocol['train_counts'])/min(self.protocol['train_counts']),
                   weight_body_reverified=False,weight_provenance='original locked cache chain only',
                   permitted_feature_splits=['train','val'],test_access=False,cache_read_seconds=time.monotonic()-t)
        write(self.public/'ASSET_AND_CACHE_AUDIT.json',audit)
        # Exact allow-list for subsequent data reads; Python/library source reads remain legal.
        allowed={str((cache/(s+'.npz')).resolve()) for s in ('train','val')}
        private=str(self.private.resolve())+os.sep
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
            path=Path(os.fsdecode(args[0])); name=path.name.lower(); resolved=str(path.resolve())
            if path.suffix.lower() in ('.jpg','.jpeg','.png','.pt','.pth','.safetensors'):
                raise RuntimeError('FORBIDDEN_IMAGE_OR_WEIGHT_ACCESS')
            if path.suffix.lower() in ('.npz','.npy') and resolved not in allowed and not resolved.startswith(private):
                raise RuntimeError('FEATURE_DATA_OUTSIDE_ALLOWLIST')
            if name in ('test.csv','test.npz','test_associations.json') or ('test' in path.parts and path.suffix in ('.csv','.npz','.json')):
                raise RuntimeError('FORBIDDEN_TEST_ACCESS')
        sys.addaudithook(guard)
        self.checks['audit']=audit['status']; return audit

    def get_current(self,label,method,mix_seed):
        # Called only by the current-arrival loop; cache reuse does not widen its inputs.
        train,labels,rows=self.data['train']; ix=np.flatnonzero(labels==label)
        ix=ix[np.argsort(np.array([rows[i]['sample_id'] for i in ix]),kind='stable')]
        z=train[ix].astype(np.float64)
        if label not in self.real:
            self.real[label]=moments(z)
            if len(z)<768:
                _,s,v=np.linalg.svd(z,full_matrices=False)
                self.span[label]=v[s>np.finfo(float).eps*max(z.shape)*s[0]]
        real=self.real[label]; key=(method,mix_seed,label)
        aug=None
        if method=='B1': aug=moments(z.copy())
        if method in ('B2','B3','B4'):
            if key not in self.aug:
                zz,record,pi,j,_,_=mixing(z,method,mix_seed,label)
                norm=np.linalg.norm(zz,axis=1)
                if method=='B2': assert np.all(norm<=np.maximum(np.linalg.norm(z[pi],axis=1),np.linalg.norm(z[j],axis=1))+1e-10)
                else: assert np.max(np.abs(norm-1))<1e-10
                self.aug[key]=moments(zz)
                for field in ('identity_component','lesion_id'):
                    ids=np.array([rows[i][field] for i in ix]); record['same_'+field+'_pair_fraction']=float(np.mean(ids[pi]==ids[j]))
                record.update(sum_sha256=digest(self.aug[key]['s']),second_moment_sha256=digest(self.aug[key]['T']))
                self.sampling.append(record)
                g,mu=geometry(z,zz,label,method,mix_seed,self.span.get(label)); self.geometries.append(g); self.means[str(key)]=mu
            aug=self.aug[key]
        elif str(key) not in self.means:
            g,mu=geometry(z,z.copy() if method=='B1' else None,label,method,mix_seed,self.span.get(label)); self.geometries.append(g); self.means[str(key)]=mu
        return real,aug

    def reference(self,seen):
        key=tuple(sorted(seen))
        if key not in self.refs:
            G=sum(self.real[c]['T']/self.real[c]['n'] for c in key)/len(key)
            R=np.stack([self.real[c]['s']/self.real[c]['n'] for c in key],1)/len(key)
            e,U=np.linalg.eigh((G+G.T)/2)
            self.refs[key]=(G,R,U[:,-16:][:,::-1],e)
        return self.refs[key]

    def spectral(self,G,R,W,res,learner,method,mix_seed,order_seed,stage):
        labels=learner.visited; base,baseR,U,_=self.reference(labels); canonical=sorted(labels)
        e=np.linalg.eigvalsh((G+G.T)/2); de=np.linalg.eigvalsh(((G-base)+(G-base).T)/2)
        row=dict(method=method,mix_seed=mix_seed,order_seed=order_seed,session=stage,solve_relative_residual=res,
                 eigenvalues_G=e.tolist(),eigenvalues_delta_B0=de.tolist(),delta_trace=float(de.sum()),
                 delta_positive_spectral_sum=float(de[de>0].sum()),delta_negative_spectral_sum=float(de[de<0].sum()),
                 delta_lambda_min=float(de[0]),delta_lambda_max=float(de[-1]),B0_basis_seen_labels=canonical,
                 B0_top16_energy=float(np.trace(U.T@G@U)),B0_complement_energy=float(np.trace(G)-np.trace(U.T@G@U)))
        for prefix,ee in [('G',e),('A',e+LAMBDA)]: row.update({prefix+'_'+k:v for k,v in eigen_diagnostics(ee).items()})
        self.spectra.append(row)
        for j,label in enumerate(labels):
            real=self.real[label]; aug=(real if method=='B1' else self.aug.get((method,mix_seed,label)))
            H,h,_=contribution(real,aug,method); H=H/learner.scale; h=h/learner.scale
            bh=baseR[:,canonical.index(label)]; proj=np.diag(U.T@H@U)
            self.projections.append(dict(method=method,mix_seed=mix_seed,order_seed=order_seed,session=stage,original_label=label,
                B0_top16_direction_second_moments=proj.tolist(),B0_top16_second_moment_sum=float(proj.sum()),
                B0_complement_second_moment=float(np.trace(H)-proj.sum()),R_top16=(U.T@h).tolist(),
                delta_R_top16=(U.T@(h-bh)).tolist(),R_complement_norm=float(np.linalg.norm(h-U@(U.T@h))),
                delta_R_complement_norm=float(np.linalg.norm((h-bh)-U@(U.T@(h-bh))))))

    def fit(self,method,mix_seed):
        print(json.dumps(dict(event='fit_start',method=method,mix_seed=mix_seed)),flush=True)
        reference_final=None
        for order_seed,order in self.orders.items():
            inc=Increment(order,768,method); current=0
            for stage,seen in enumerate((4,6,8)):
                for label in order[current:seen]:
                    real,aug=self.get_current(label,method,mix_seed)
                    inc.arrive(label,real,aug)
                    self.arrivals.append(dict(method=method,mix_seed=mix_seed,order_seed=order_seed,session=stage,arrived_label=label,visited=inc.visited.copy()))
                current=seen; G,R=inc.system(); W,res=solve(G,R); self.solves+=1
                # Independent direct combination of exactly this stage's arrived class statistics.
                parts=[contribution(self.real[c],self.real[c] if method=='B1' else self.aug.get((method,mix_seed,c)),method) for c in order[:seen]]
                scale=sum(p[2] for p in parts)
                close(G,sum(p[0] for p in parts)/scale); close(R,np.stack([p[1] for p in parts],1)/scale)
                if method=='B1':
                    bg,br,_,_=self.reference(order[:seen]); br=br[:,[sorted(order[:seen]).index(c) for c in order[:seen]]]
                    close(G,bg); close(R,br)
                    with np.load(self.private/f'B0_0_{order_seed}_s{stage}.npz') as f: close(W,f['W'])
                inc.save(self.private/'active_state.npz')
                restored=Increment.restore(self.private/'active_state.npz'); close(restored.H,inc.H); inc=restored
                val,y,rows=self.data['val']; ix=np.flatnonzero(np.isin(y,order[:seen])); inverse={c:j for j,c in enumerate(order[:seen])}
                raw=val[ix].astype(np.float64)@W
                p=dict(raw=raw,y=np.array([inverse[int(c)] for c in y[ix]]),original=y[ix],order=np.array(order),
                       ids=np.array([rows[i]['sample_id'] for i in ix]),component=np.array([rows[i]['identity_component'] for i in ix]),lesion=np.array([rows[i]['lesion_id'] for i in ix]))
                assert p['ids'].tolist()==sorted(p['ids'].tolist())
                if method=='B1':
                    with np.load(self.private/f'B0_0_{order_seed}_s{stage}.npz') as f:
                        close(raw,f['raw']); assert np.array_equal(raw.argmax(1),f['raw'].argmax(1))
                np.savez_compressed(self.private/f'{method}_{mix_seed}_{order_seed}_s{stage}.npz',W=W,**p)
                m,pc,er=analyze_unit(p,method,order_seed,stage)
                other=raw.copy(); other[np.arange(len(raw)),p['y']]=-np.inf
                margin=raw[np.arange(len(raw)),p['y']]-other.max(1)
                for record in [m]+pc+er: record.update(split='val',predictor='G1_analytic_all_seen',mix_seed=mix_seed)
                m['raw_correct_class_margin_quantiles']=np.quantile(margin,[0,.05,.25,.5,.75,.95,1]).tolist()
                for record in pc:
                    mask=p['y']==record['head_index']; record['n_correct']=int(np.count_nonzero(raw[mask].argmax(1)==p['y'][mask]))
                    record['raw_correct_class_margin_quantiles']=np.quantile(margin[mask],[0,.05,.25,.5,.75,.95,1]).tolist()
                for c in (6,7): m[f'label_{c}_recall']=next((r['recall'] for r in pc if r['original_label']==c),None)
                assert abs(m['balanced_accuracy']-np.mean([r['recall'] for r in pc]))<1e-10
                self.metrics.append(m); self.per_class+=pc; self.errors+=er
                self.spectral(G,R,W,res,inc,method,mix_seed,order_seed,stage)
                if stage==2:
                    canon=np.argsort(order); final=(G,R[:,canon],W[:,canon],np.array(order)[raw.argmax(1)])
                    if reference_final is None:
                        reference_final=tuple(x.copy() for x in final)
                        if method!='B1': inc.save(self.private/f'{method}_{mix_seed}_final_state.npz')
                    else:
                        for a,b in zip(reference_final[:3],final[:3]): close(a,b)
                        assert np.array_equal(reference_final[3],final[3]),'FINAL_ORDER_PREDICTION_MISMATCH'
                self.resources()
        self.final[method,mix_seed]=True
        self.flush()
        # Augmented sufficient statistics are no longer needed after this method/seed.
        self.aug.clear()
        print(json.dumps(dict(event='fit_complete',method=method,mix_seed=mix_seed,rows=len(self.metrics))),flush=True)

    def flush(self):
        for name,rows in [('val_metrics',self.metrics),('val_per_class_metrics',self.per_class),('error_decomposition',self.errors),
                          ('spectral_diagnostics',self.spectra),('class_geometry_diagnostics',self.geometries),('class_spectral_projections',self.projections)]:
            if rows: csvwrite(self.public/(name+'.csv'),rows)
        write(self.public/'SAMPLING_AUDIT.json',dict(records=self.sampling,old_classes_resampled=False,local_rng_only=True))
        write(self.private/'arrival_access_audit.json',self.arrivals)
        np.savez_compressed(self.private/'class_means.npz',**self.means)

    def p0(self):
        audit=self.audit(); self.checks['toy']=engineering(self.private,ROOT/'tools/frozen_medical_v3.py')
        t=time.monotonic(); z,y,_=self.data['train']; head=z[y==0].astype(np.float64)
        augmented=mixing(head,'B4',41001,0)[0]; s=moments(augmented)
        bench=time.monotonic()-t; del head,augmented,s
        # Conservative probe-based bound includes eigensystems and report overhead.
        forecast=30*bench+160*self.checks['toy']['seconds']+300
        assert forecast<7200, 'BLOCKED_PROJECTED_CPU_BUDGET'
        self.checks['budget']=dict(cache_read_seconds=audit['cache_read_seconds'],head_class_aug_stats_seconds=bench,
                                  projected_seconds=forecast,projected_new_bytes=120*1024**2,threads=4,workers=1)
        self.resources(); write(self.public/'ENGINEERING_G1.json',self.checks)
        print(json.dumps(dict(event='P0_PASS',budget=self.checks['budget'])),flush=True)

    def p1(self):
        for method in ('R0','B0','B1'): self.fit(method,0)
        history=list(csv.DictReader((self.cache/'frozen_val_metrics.csv').open()))
        classes=list(csv.DictReader((self.cache/'frozen_per_class_metrics.csv').open()))
        maxdiff=0
        for row in self.metrics:
            if row['method']!='B0': continue
            old=next(r for r in history if r['classifier']=='CBRidge' and int(r['order_seed'])==row['order_seed'] and int(r['session'])==row['session'])
            for metric in ('balanced_accuracy','accuracy','macro_f1','head_rank2','mid_rank4','tail_rank2','old_macro_recall','current_macro_recall'):
                if row[metric] is None: assert old[metric]==''
                else:
                    d=abs(row[metric]-float(old[metric])); maxdiff=max(maxdiff,d); assert d<1e-10,'BLOCKED_B0_REPRODUCIBILITY'
        for row in self.per_class:
            if row['method']!='B0':continue
            old=next(r for r in classes if r['classifier']=='CBRidge' and int(r['order_seed'])==row['order_seed'] and int(r['session'])==row['session'] and int(r['original_label'])==row['original_label'])
            assert abs(row['recall']-float(old['recall']))<1e-10
        assert len(self.metrics)==27 and len(self.per_class)==162
        self.checks.update(P1='PASS',baseline_max_metric_difference_pp=maxdiff,B1_G_R_W_scores_predictions='PASS',stage_access='PASS',resume_every_stage='PASS')
        write(self.public/'ENGINEERING_G1.json',self.checks)
        files=['tools/isic_g1_controlled_mixup.py','tools/report_isic_g1.py','tools/report_locked_holdout_r1.py','tools/frozen_medical_v3.py','tests/test_isic_g1_controlled_mixup.py','exps/isic_g1_protocol.json']
        lock=dict(source_commit=self.cfg['source_commit'],sha256={p:sha(ROOT/p) for p in files},protocol='G1',phase='after_P1_before_P2',runtime=dict(numpy=np.__version__,scipy=scipy.__version__,python=sys.version),**COUNTERS)
        write(self.public/'CODE_LOCK_G1.json',lock); write(self.public/'ENGINEERING_PASS_G1.json',dict(status='PASS',rows=27,baseline_max_difference_pp=maxdiff,code_lock_sha256=sha(self.public/'CODE_LOCK_G1.json')))
        print('P1_PASS_CODE_LOCKED',flush=True)

    def all(self):
        self.p0(); self.p1()
        for method in ('B2','B3','B4'):
            for seed in (41001,41002,41003): self.fit(method,seed)
        assert len(self.metrics)==108 and len(self.per_class)==648
        # Pairing and base uniforms must match across all variants for every class/seed.
        indexed={(r['method'],r['mix_seed'],r['original_label']):r for r in self.sampling}
        for seed in (41001,41002,41003):
            for label in range(8):
                rs=[indexed[m,seed,label] for m in ('B2','B3','B4')]
                for field in ('pair_sha256','uniform_sha256'): assert len({r[field] for r in rs})==1
                assert rs[0]['gamma_sha256']==rs[1]['gamma_sha256']
        write(self.public/'P2_COMPLETE.json',dict(status='PASS',logical_rows=108,per_class_rows=648,method_randomness_final_fits=12,actual_formal_solve_calls=self.solves,final_order_invariance=True,stream_batch_G_R_equivalence=True))
        from report_isic_g1 import complete
        complete(self.out)
        report=self.resources(); report.update(**COUNTERS,core_logical_metric_rows=108,core_logical_per_class_rows=648,
           continued_increment_bytes_max=768*768*8+768*8*8+8*8,statistics_archives='11 deduplicated final states plus one active state; no synthetic feature archives',
           formal_augmented_stat_generations=len(self.sampling),separate_P0_head_probe_generations=1,
           source_commit=self.cfg['source_commit'],technical_status='COMPLETE_G1_P0_P3')
        write(self.public/'RESOURCE_REPORT.json',report)
        from report_isic_g1 import render_report
        render_report(self.public)
        write(self.public/'COMPLETION_AUDIT.json',dict(status='COMPLETE_G1_P0_P3',**COUNTERS,core_logical_metric_rows=108,core_logical_per_class_rows=648))
        print(json.dumps(dict(event='COMPLETE_G1_P0_P3',resources=report)),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['audit','engineering','fit','report','run-all'],nargs='?',default='run-all')
    parser.add_argument('--split',choices=['train','val'],default='val')
    args=parser.parse_args(); config=read(os.environ['P16_CONFIG'])
    with threadpool_limits(limits=4):
        run=Run(config)
        try:
            if args.phase=='run-all' or args.phase=='fit': run.all()
            elif args.phase=='audit': run.audit()
            elif args.phase=='engineering': run.p0()
            else:
                from report_isic_g1 import complete,render_report
                complete(run.out); render_report(run.public)
        except Exception as e:
            write(run.public/'BLOCKED.json',dict(status='BLOCKED',reason=str(e),type=type(e).__name__,completed_metric_rows=len(run.metrics),**COUNTERS))
            raise


if __name__=='__main__': main()
