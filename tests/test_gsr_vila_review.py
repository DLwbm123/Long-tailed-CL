"""Independent protocol regressions for commit 74017f1.

These tests intentionally fail against the reviewed commit.  They use only
synthetic tensors/configuration, never patient images/features/checkpoints.
They are acceptance tests for a corrected implementation, not a patch.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shared.frozen_dual_features import build_joint_feature, label_free_forward, validate_encoder_lock
from shared.dual_moment_bank import DualMomentBank
from route_a.semantic_prior import component_reliability
from route_a.spectral_prior_ridge import a0_to_a8, spectral_prior_ridge
from route_a.run_ct13 import validate_config as validate_a
from route_b.run_ct14 import validate_config as validate_b
from route_b.anchor_drift import fit_anchor_drift, torch_anchor_drift
from route_b.affine_moment_transport import homogeneous_matrix, transport_old_bank
from route_b.margin_memory_loss import memory_margin_loss, current_query_risk

DT = torch.float64
ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return json.loads((ROOT / 'configs' / 'gsr_vila' / name).read_text())


def test_memory_margin_uses_covariance_off_diagonal():
    W = torch.eye(2, dtype=DT)
    M = torch.eye(2, dtype=DT)
    C = torch.tensor([[1., .9], [.9, 1.]], dtype=DT)
    loss_corr = memory_margin_loss(W, M, C)
    loss_diag = memory_margin_loss(W, M, torch.diag(C.diag()))
    # q=(1,-1): q^T C q=.2, while diagonal-only variance=2.
    assert abs(float(loss_corr-loss_diag)) > 1e-4


def test_memory_margin_includes_zero_logit_for_log_one_plus():
    W = torch.eye(2, dtype=DT)
    M = 2*torch.eye(2, dtype=DT)
    C = torch.zeros((2, 2), dtype=DT)
    got = memory_margin_loss(W, M, C)
    expected = .1*torch.nn.functional.softplus(torch.tensor((.1+.5*np.sqrt(1e-8)-2)/.1,dtype=DT))
    torch.testing.assert_close(got, expected, atol=1e-9, rtol=1e-8)


def test_one_old_class_still_has_new_class_competitors():
    W = torch.tensor([[1., 0.]], dtype=DT)
    M = torch.tensor([[-1.]], dtype=DT)
    C = torch.zeros((1, 1), dtype=DT)
    assert float(memory_margin_loss(W,M,C)) > 1.


def test_perfect_one_hot_query_has_zero_squared_risk():
    logits = torch.eye(2, dtype=DT)
    labels = torch.tensor([0,1])
    assert float(current_query_risk(logits, labels)) == pytest.approx(0.,abs=1e-12)


def test_query_risk_is_class_balanced_squared_error():
    logits = torch.tensor([[0.,0.],[0.,0.],[0.,1.]],dtype=DT)
    labels = torch.tensor([0,0,1])
    # Class0 risk=1, class1 risk=0; equal class weighting gives .5.
    assert float(current_query_risk(logits, labels)) == pytest.approx(.5,abs=1e-12)


def test_homogeneous_bank_has_homogeneous_dimensions_and_unit_bottom_means():
    bank = DualMomentBank(dim_a=1,dim_u=1)
    bank.add_class(0,np.array([[1.,0.]]),task=1)
    bank.add_class(1,np.array([[0.,1.]]),task=1)
    Sbar, Mbar = bank.homogeneous()
    assert Sbar.shape == (3,3) and Mbar.shape == (3,2)
    np.testing.assert_allclose(Sbar[-1,-1],2)
    np.testing.assert_allclose(Mbar[-1],1)


def test_current_normalized_bank_transport_matches_explicit_samples():
    # This deliberately exercises the exact G,R contract used in the PR's own test.
    X=np.array([[1.,0.],[0.,1.]])
    bank=DualMomentBank(dim_a=1,dim_u=1)
    bank.add_class(0,X[:1],task=1);bank.add_class(1,X[1:],task=1)
    G,R=bank.homogeneous()
    B=np.zeros((1,1));b=np.array([.05])
    gotG,gotR=transport_old_bank(G,R,B,b,dim_a=1,dim_u=1)
    Y=X.copy();Y[:,0]+=.05
    np.testing.assert_allclose(gotG,Y.T@Y/2,atol=1e-12)
    np.testing.assert_allclose(gotR,Y.T/2,atol=1e-12)


def test_singleton_components_do_not_persist_individual_joint_features():
    X=np.array([[1.,0.],[0.,1.]])
    bank=DualMomentBank(dim_a=1,dim_u=1)
    bank.add_class(0,X,task=1,component_ids=['g0','g1'])
    stored=bank.state_dict()['classes']['0']
    # Learning state must reduce the class's component distribution, not retain
    # all per-component full joint vectors/outer products.
    assert not stored.get('component_means') and not stored.get('component_seconds')


def test_component_reliability_inputs_count_all_components_of_each_class():
    bank=DualMomentBank(dim_a=1,dim_u=2)
    for c in (0,1):
        bank.add_class(c,np.eye(3)[:2],task=1,component_ids=['g0','g1'])
    means,counts=bank.component_reliability_inputs('g0')
    assert counts == {0:2,1:2}


def test_semantic_reliability_is_invariant_to_original_label_ids():
    T=np.eye(2)
    means={4:np.tile([1.,0.],(3,1)),7:np.tile([0.,1.],(3,1))}
    r,gamma=component_reliability(T,means,{4:3,7:3},class_ids=[4,7],a_scale=1.)
    np.testing.assert_allclose(r,np.ones(2),atol=1e-7)


def test_semantic_reliability_uses_within_class_difference_variance():
    T=np.eye(2)
    means={0:np.tile([1.,0.],(2,1)),1:np.tile([0.,1.],(2,1))}
    r,_=component_reliability(T,means,{0:2,1:2},class_ids=[0,1],a_scale=1.)
    # All component means within a class identical: Cc=Cpool=0 and m=1.
    expected=np.ones(2)/(1.+1e-8)
    np.testing.assert_allclose(r,expected,atol=1e-10)


def test_u_stats_restore_unscaled_unit_clip_coordinates():
    f=build_joint_feature(np.array([[1.],[2.]]),np.eye(2))
    bank=DualMomentBank(dim_a=1,dim_u=2)
    bank.add_class(0,f.h[:1],task=1);bank.add_class(1,f.h[1:],task=1)
    Gu,Ru=bank.u_stats()
    np.testing.assert_allclose(Gu,f.u.T@f.u/2,atol=1e-12)
    np.testing.assert_allclose(Ru,f.u.T/2,atol=1e-12)


def test_affine_numpy_bridge_cannot_be_used_as_training_gradient_path():
    v=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],dtype=DT)
    d=torch.tensor([[.01,.02],[.03,.01],[-.01,.02]],dtype=DT,requires_grad=True)
    B,b=torch_anchor_drift(v,d)
    H=homogeneous_matrix(B,b,dim_a=2,dim_u=2)
    assert isinstance(H,torch.Tensor) and H.requires_grad


def test_translation_fit_is_class_balanced_not_sample_balanced():
    # Three observations of class0 and one of class1; both conditional=false.
    # Class0 drift .02 and class1 drift -.02 have class-balanced mean zero.
    v=np.array([[0.],[0.],[0.],[1.]])
    d=np.array([[.02],[.02],[.02],[-.02]])
    _,b,_=fit_anchor_drift(v,d,conditional=False)
    np.testing.assert_allclose(b,np.zeros(1),atol=1e-12)


def test_feature_builder_accepts_differentiable_feature_path():
    a=torch.ones((2,3),dtype=DT,requires_grad=True)
    u=torch.ones((2,2),dtype=DT)
    f=build_joint_feature(a,u)
    assert isinstance(f.h,torch.Tensor) and f.h.requires_grad


def test_lock_validator_rejects_arbitrary_non_digest_strings():
    with pytest.raises(ValueError):
        validate_encoder_lock(dict(model_name='not-clip',pretrained='wrong',weights_sha256='bad',preprocess_sha256='bad'))


def test_route_configs_preserve_original_seed_order_matrix():
    for name in ('CONFIG_A_RASP.json','CONFIG_B_ACTM.json'):
        assert config(name)['seeds']==[1993,1994,1995]


def test_config_rejects_empty_seeds_and_extra_tasks():
    cfg=config('CONFIG_A_RASP.json');cfg['seeds']=[];cfg['tasks']=[1,2,3,4,99]
    with pytest.raises(ValueError):validate_a(cfg)


def test_config_rejects_duplicate_variants_and_excess_epochs():
    cfg=config('CONFIG_B_ACTM.json');cfg['variants'].append('B11');cfg['epochs']=999;cfg['tasks']=[5]
    with pytest.raises(ValueError):validate_b(cfg)


@pytest.fixture(scope='module')
def all_readouts():
    # Full intended dimensions, realizable positive per-class moments,
    # no random image features or patient data.
    d=2048;K=2
    diag=np.r_[np.full(1536,.5/1536),np.full(512,.5/512)]
    G=np.diag(diag);M=np.zeros((d,K))
    M[0,0]=.01;M[1,1]=.01;M[1536,0]=.02;M[1537,1]=.02
    text=np.eye(512)[:,:K]
    outputs=a0_to_a8(K*G,M,text,V=np.zeros_like(M),gamma=0.,ncomp=np.ones(K))
    return outputs,diag,M


def test_a0_a1_match_original_unscaled_branch_ridge(all_readouts):
    outputs,diag,M=all_readouts
    # Compare predictions on actual joint subblock x against prediction on
    # original unit block a=sqrt(2)*x.  Equality must hold for each branch.
    for key,sl in [('A0',slice(0,1536)),('A1',slice(1536,None))]:
        expected_on_scaled_input=np.sqrt(2.)*M[sl]/(2*(diag[sl,None])+.001)
        expected_on_scaled_input /= 2
        np.testing.assert_allclose(outputs[key],expected_on_scaled_input,rtol=1e-10,atol=1e-10)


def test_apart_requires_and_concatenates_both_prelogits_branches():
    main = np.zeros((2, 768), dtype=np.float64)
    few = np.ones((2, 768), dtype=np.float64)
    f = label_free_forward(np.zeros((2, 1)), lambda _: {'pre_logits': main, 'pre_logits_few': few},
                           lambda _: np.ones((2, 2)))
    assert f.a.shape == (2, 1536) and np.allclose(f.a[:, :768], np.asarray(main) / np.linalg.norm(np.c_[main, few], axis=1, keepdims=True))
    with pytest.raises(ValueError, match='BLOCKED_APART_PRELOGITS_CONTRACT'):
        label_free_forward(np.zeros((1, 1)), lambda _: {'pre_logits': np.ones((1, 6))},
                           lambda _: np.ones((1, 2)))


def test_a5_zero_semantic_scale_reverts_to_a2(all_readouts):
    d=2048;K=2
    diag=np.r_[np.full(1536,.5/1536),np.full(512,.5/512)]
    G=np.diag(diag);M=np.zeros((d,K));text=np.eye(512)[:,:K]
    outputs=a0_to_a8(K*G,M,text,V=np.zeros_like(M),gamma=0.,a_scale=0.)
    np.testing.assert_allclose(outputs['A5'],outputs['A2'],rtol=1e-10,atol=1e-10)


def test_a3_is_not_merely_text_prototype_instead_of_cse_readout(all_readouts):
    outputs,_,_=all_readouts
    # A3 is input-dependent: it must bundle/use A2 proposals + topK fusion,
    # not masquerade a lone 512xK text prototype as the completed classifier.
    assert not isinstance(outputs['A3'],np.ndarray) or outputs['A3'].shape[0] != 512


# Positive controls: the review does not imply every mathematical primitive fails.
def test_positive_joint_feature_normalization():
    rng=np.random.default_rng(21)
    f=build_joint_feature(rng.normal(size=(5,4)),rng.normal(size=(5,2)))
    np.testing.assert_allclose(np.linalg.norm(f.h,axis=1),1.,atol=1e-14)


def test_positive_spectral_prior_nonzero_gamma_normal_equations():
    rng=np.random.default_rng(22);X=rng.normal(size=(20,6));K=3
    G=X.T@X/20;M=rng.normal(size=(6,K));V=rng.normal(size=(6,K));gam=np.array([0.,.003,.02])
    W,diag=spectral_prior_ridge(K*G,M,V,gam)
    for c in range(K):
        expected=np.linalg.solve(G+.001*np.eye(6)+gam[c]*diag['P'],M[:,c]/K+gam[c]*diag['P']@V[:,c])
        np.testing.assert_allclose(W[:,c],expected,atol=1e-11)


def test_positive_a8_matches_centered_ridge(all_readouts):
    outputs,diag,M=all_readouts;d=len(diag);K=M.shape[1]
    G=np.diag(diag);m=M.mean(1);yb=np.full(K,1/K)
    W=np.linalg.solve(G-np.outer(m,m)+.001*np.eye(d),M/K-np.outer(m,yb))
    expected=np.vstack([W,yb-m@W])
    np.testing.assert_allclose(outputs['A8'],expected,atol=1e-10,rtol=1e-10)


def test_positive_anchor_torch_matches_numpy_equal_weight_fit():
    rng=np.random.default_rng(23);v=rng.normal(size=(12,3));d=rng.normal(size=(12,2))*.01
    B,b,_=fit_anchor_drift(v,d)
    Bt,bt=torch_anchor_drift(torch.tensor(v,dtype=DT),torch.tensor(d,dtype=DT,requires_grad=True))
    np.testing.assert_allclose(Bt.detach(),B,atol=1e-12)
    np.testing.assert_allclose(bt.detach(),b,atol=1e-12)
