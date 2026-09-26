"""Synthetic identities only. Run: python -m unittest discover -s reference -v"""
import unittest
import numpy as np
from tools.rfvila_reference import (MomentBank, normalize_rows, projections, features, batch_ridge,
                          weighted_design, group_folds, select_lambda_cv, pick_lambda,
                          cse, predict, metrics)

class MathChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(87026)
        self.a = normalize_rows(rng.normal(size=(24, 5)))
        self.u = normalize_rows(rng.normal(size=(24, 3)))
        self.ra, self.ru = projections(5, 3, 12, 67101)
        self.y = np.array([8]*10 + [2]*8 + [9]*6)
        self.ids = [8, 2, 9]
        self.x = features(self.a, self.u, self.ra, self.ru, "J.RF")

    def bank(self, x=None, y=None, ids=None):
        x = self.x if x is None else x; y = self.y if y is None else y
        ids = self.ids if ids is None else ids
        b = MomentBank(x.shape[1])
        for c in ids: b.append_class(c, x[y == c])
        return b

    def test_01_unit_rows(self):
        np.testing.assert_allclose(np.linalg.norm(self.a, axis=1), 1)

    def test_02_zero_rejected(self):
        with self.assertRaises(ValueError): normalize_rows(np.zeros((1,3)))

    def test_03_projection_determinism(self):
        ra, ru = projections(5,3,12,67101)
        np.testing.assert_array_equal(ra,self.ra); np.testing.assert_array_equal(ru,self.ru)

    def test_04_projection_seeds_differ(self):
        ra, _ = projections(5,3,12,67102)
        self.assertFalse(np.array_equal(ra,self.ra))

    def test_05_no_global_rng_consumption(self):
        np.random.seed(19); expected=np.random.normal(size=5)
        np.random.seed(19); projections(5,3,12,67101)
        np.testing.assert_array_equal(expected,np.random.normal(size=5))

    def test_06_joint_block_equals_dense(self):
        h=np.concatenate([self.a,self.u],1)/np.sqrt(2)
        dense=np.sqrt(2/12)*np.maximum(h@np.vstack([self.ra,self.ru]),0)
        np.testing.assert_allclose(self.x,dense,atol=1e-15)

    def test_07_split_capacity(self):
        self.assertEqual(features(self.a,self.u,self.ra,self.ru,'J.SPLIT').shape,self.x.shape)

    def test_08_linear_random_is_linear(self):
        z=features(self.a,self.u,self.ra,self.ru,'J.RPLINEAR')
        h=features(self.a,self.u,self.ra,self.ru,'J.LIN')
        np.testing.assert_allclose(z,h@np.vstack([self.ra,self.ru])/np.sqrt(12),atol=1e-15)

    def test_09_batch_equals_incremental(self):
        w=self.bank().solve(.01)
        np.testing.assert_allclose(w,batch_ridge(self.x,self.y,self.ids,.01),rtol=1e-10,atol=1e-12)

    def test_10_dual_equals_primal(self):
        np.testing.assert_allclose(batch_ridge(self.x,self.y,self.ids,.01),
                                   batch_ridge(self.x,self.y,self.ids,.01,dual=True),rtol=1e-10,atol=1e-12)

    def test_11_row_permutation(self):
        perm=np.random.default_rng(3).permutation(len(self.y))
        np.testing.assert_allclose(self.bank().solve(.01),self.bank(self.x[perm],self.y[perm]).solve(.01),atol=1e-12)

    def test_12_class_duplication_invariance(self):
        x=np.concatenate([self.x,self.x[self.y==8]],0); y=np.concatenate([self.y,self.y[self.y==8]])
        np.testing.assert_allclose(self.bank().solve(.01),self.bank(x,y).solve(.01),atol=1e-12)

    def test_13_k_lambda_is_required(self):
        b=self.bank(); good=b.solve(.01)
        bad=np.linalg.solve(b.S+.01*np.eye(b.dim),b.Q)
        self.assertGreater(np.linalg.norm(good-bad),1e-3)

    def test_14_duplicate_class_rejected(self):
        b=self.bank()
        with self.assertRaises(ValueError): b.append_class(8,self.x[:2])

    def test_15_column_order(self):
        b1=self.bank(); b2=self.bank(ids=[9,8,2])
        np.testing.assert_allclose(b1.solve(.01),b2.solve(.01)[:,[1,2,0]],atol=1e-12)

    def test_16_feature_scale_lambda_equivalence(self):
        w=batch_ridge(self.x,self.y,self.ids,.01)
        w2=batch_ridge(3*self.x,self.y,self.ids,.09)
        np.testing.assert_allclose(self.x@w,3*self.x@w2,atol=1e-12)

    def test_17_corrected_woodbury(self):
        b=MomentBank(self.x.shape[1]); b.append_class(8,self.x[self.y==8]); lam=.01
        old=np.linalg.inv(b.S+lam*np.eye(b.dim))
        shifted=np.linalg.solve(np.eye(b.dim)+lam*old,old)
        z=self.x[self.y==2]/np.sqrt(np.sum(self.y==2))
        new=shifted-shifted@z.T@np.linalg.solve(np.eye(len(z))+z@shifted@z.T,z@shifted)
        b.append_class(2,self.x[self.y==2])
        np.testing.assert_allclose(new@b.Q,b.solve(lam),rtol=1e-10,atol=1e-12)

    def test_18_relu_moments_not_recoverable(self):
        x=np.array([-1.,1.]); y=np.array([-np.sqrt(2),0.,0.,np.sqrt(2)])
        self.assertAlmostEqual(x.mean(),y.mean()); self.assertAlmostEqual((x*x).mean(),(y*y).mean())
        self.assertGreater(abs(np.maximum(x,0).mean()-np.maximum(y,0).mean()),.1)

    def test_19_group_fold_no_leak(self):
        y=np.repeat([8,2],12); c=np.array([f'{a}-{i//2}' for a in [8,2] for i in range(12)])
        f=group_folds(y,c)
        self.assertEqual(set(f),{0,1,2})
        for g in set(c): self.assertEqual(len(set(f[c==g])),1)
        for i in set(f): self.assertEqual(set(y[f==i]),{8,2})

    def test_20_singleton_fallback(self):
        f=group_folds([8,8,2],['a','b','c']); self.assertIsNone(f)

    def test_21_cross_label_group_fallback(self):
        self.assertIsNone(group_folds([8,8,2,2],['x','a','x','b']))

    def test_22_lambda_tie_larger(self):
        self.assertEqual(pick_lambda([.001,.01,.1],[2,1,1]),.1)

    def test_23_cv_uses_only_current_supplied_rows(self):
        y=np.repeat([8,2],8); x=self.x[:16]; c=np.array([f'g{i}' for i in range(16)])
        lam, log=select_lambda_cv(x,y,c,[8,2],[.001,.01])
        self.assertIn(lam,[.001,.01]); self.assertEqual(log['folds'],3)

    def test_24_alpha_zero(self):
        s=self.x[:,:3]; np.testing.assert_array_equal(cse(s,s,self.ids,alpha=0),s)

    def test_25_sparse_cse_outside_unchanged(self):
        s=np.array([[.9,.8,.1]]); v=np.array([[.1,.2,10.]])
        out=cse(s,v,self.ids,top_k=2)
        np.testing.assert_allclose(out,[[1.,1.,.1]])

    def test_26_quarter_matches_author_argmax(self):
        s=self.x[:,:3]; v=np.flip(s,1)
        x=cse(s,v,self.ids,alpha=.25,top_k=2)
        mask=cse(np.zeros_like(s),np.zeros_like(s),self.ids) # shape-only
        for i in range(len(s)):
            k=np.lexsort((np.array(self.ids),-s[i]))[:2]; mask[i,k]=v[i,k]
        np.testing.assert_array_equal(predict(x,self.ids),predict(.8*s+.2*mask,self.ids))

    def test_27_ties_original_ids(self):
        np.testing.assert_array_equal(predict(np.ones((2,3)),self.ids),[2,2])

    def test_28_macro_f1_not_ba(self):
        v=metrics([0,0,0,1],[0,0,0,0],[0,1])
        self.assertAlmostEqual(v['ba'],.5); self.assertAlmostEqual(v['macro_f1'],3/7)

    def test_29_zero_recall_set_not_count(self):
        a=metrics([0,1,2],[0,0,2],[0,1,2]); b=metrics([0,1,2],[0,1,0],[0,1,2])
        self.assertEqual(len(a['zero_recall_ids']),len(b['zero_recall_ids']))
        self.assertNotEqual(a['zero_recall_ids'],b['zero_recall_ids'])

    def test_30_nonfinite_rejected(self):
        with self.assertRaises(ValueError): self.bank(np.full_like(self.x,np.nan))

    def test_31_new_task_three_classes(self):
        rng=np.random.default_rng(5); b=MomentBank(6); xx=[]; yy=[]
        for c,n in zip([4,0,3,6,2],[3,4,2,5,3]):
            x=rng.normal(size=(n,6)); b.append_class(c,x); xx.append(x); yy.extend([c]*n)
        np.testing.assert_allclose(b.solve(.02),batch_ridge(np.vstack(xx),np.array(yy),[4,0,3,6,2],.02),atol=1e-12)

    def test_32_energy_expectation(self):
        ra,ru=projections(5,3,16384,67101)
        for kind in ['F.RF','J.RF','J.SPLIT','J.RPLINEAR']:
            z=features(self.a,self.u,ra,ru,kind)
            self.assertLess(abs(float(np.mean(np.sum(z*z,axis=1)))-1),.05)

if __name__=='__main__': unittest.main(verbosity=2)
