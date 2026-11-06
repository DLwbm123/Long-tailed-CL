import copy
import itertools
import unittest
import numpy as np
import torch
from ..memory import from_views,join
from ..learner import Learner,optimizer,learn
from ..contracts import VisibleState,DataAccessLedger,token_keys,validate_candidate
from ..state_features import split_half,describe,action_features
from ..risk_transport import geometry,transport
from ..anchor_candidates import generate,anchors
from ..branch_env import Snapshot,PulseEnvironment,paired_learn
from ..paired_labels import evaluate,paired,utility,near_optimal,Counts
from ..policy import ResidualPolicy,verifier_prior,sft_loss,ppo_loss
from ..verifier import Verifier,FitNormalizer
from ..train_controller import fingerprint,train_verifier,train_sft,collect_trajectories,train_pg,train_more_sup,shuffle_rewards
from ..evaluate_controller import prepare,select


def fixture(C=3,n=8,capacity=24,seed=12):
    torch.manual_seed(seed)
    z=torch.nn.functional.normalize(torch.randn(C*n,6,512),dim=-1)
    bank=from_views(z,torch.arange(C*n),torch.arange(C*n)//n)
    model=Learner(seed);model.trained_groups=set();model.arrive(bank,C)
    opt=optimizer(model);counter={}
    learn(model,opt,bank,2,seed,counter,None,8,'engineering')
    groups={i:str(i) for i in range(C*n)}
    model.trained_groups=set(groups.values())
    access=DataAccessLedger({i:0 for i in groups},groups,{str(i):'S' for i in groups},0)
    state=VisibleState(bank,None,(n,)*C,(0,)*C,0,capacity,groups,True,access)
    return state,model,opt,counter


class MathematicalTests(unittest.TestCase):
    def test_odd_even_scale(self):
        for n,value in [(5,12/25),(6,.5)]:
            r=split_half(torch.randn(n,8),list(range(n)),True)
            self.assertAlmostEqual(r['chi'],value)

    def test_missing_group_evidence(self):
        z=torch.randn(8,8)
        self.assertFalse(split_half(z,None,False)['available'])
        self.assertFalse(split_half(z,[0]*8,True)['available'])
        self.assertFalse(split_half(torch.zeros(8,8),list(range(8)),True)['available'])

    def test_same_patient_more_images(self):
        z=torch.randn(6,8);a=split_half(z,list(range(6)),True)
        b=split_half(z.repeat_interleave(3,0),[i for i in range(6) for _ in range(3)],True)
        self.assertEqual(b['n'],6);self.assertAlmostEqual(a['value'],b['value'],places=6)

    def test_transport_constraints(self):
        C,S=geometry(torch.randn(7,8,dtype=torch.double));r=transport(torch.randn(7,dtype=torch.double),C,S)
        self.assertTrue(torch.allclose(S.sum(1),torch.ones(7,dtype=torch.double)))
        self.assertTrue(torch.allclose(r['Pi'].sum(0),torch.ones(7,dtype=torch.double)/7))
        self.assertAlmostEqual(float(r['q'].sum()),1.)
        self.assertTrue(bool((r['w']>=0).all()));self.assertLessEqual(float(r['w'].sum()),1.000001)

    def test_all_negative_and_singleton(self):
        for n in [1,5]:
            C,S=geometry(torch.randn(n,8));r=transport(-torch.ones(n),C,S)
            self.assertTrue(torch.equal(r['w'],torch.zeros(n)))
            self.assertTrue(torch.allclose(r['p'],torch.ones(n)/n))

    def test_signed_before_relu(self):
        C=torch.zeros(3,3,dtype=torch.double);S=torch.ones(3,3,dtype=torch.double)/3
        d=torch.tensor([1.,-2.,-2.],dtype=torch.double)
        self.assertTrue(torch.equal(transport(d,C,S)['w'],torch.zeros(3,dtype=torch.double)))
        self.assertTrue(bool((S@d.relu()>0).all()))
        self.assertTrue(bool((transport(torch.ones(3,dtype=torch.double),C,S)['w']>0).all()))

    def test_transport_autograd_and_finite_difference(self):
        C,S=geometry(torch.randn(4,8,dtype=torch.double))
        d=torch.tensor([.3,.5,.6,.8],dtype=torch.double,requires_grad=True)
        r=transport(d,C,S,tau=.7);g=torch.autograd.grad(r['value'],d)[0]
        self.assertTrue(torch.allclose(g,r['w'],atol=1e-10))
        for j in range(4):
            plus=d.detach().clone();minus=plus.clone();plus[j]+=1e-6;minus[j]-=1e-6
            fd=(transport(plus,C,S,tau=.7)['value']-transport(minus,C,S,tau=.7)['value'])/2e-6
            self.assertAlmostEqual(float(fd),float(g[j]),places=7)

    def test_probability_and_zero_residual(self):
        pi=ResidualPolicy();x=torch.randn(3,14);a=torch.randn(6,10);q=verifier_prior(torch.randn(6))
        p=pi.log_probs(x,a,q).exp()
        self.assertTrue(torch.allclose(p,q,atol=1e-7));self.assertTrue(bool((p>0).all()))
        self.assertAlmostEqual(float(p.detach().sum()),1.,places=6)
        self.assertEqual(float(pi.log_probs(x,a[:1],verifier_prior(torch.zeros(1))).detach().exp()),1.)

    def test_sampling_matches_enumeration(self):
        q=verifier_prior(torch.tensor([1.,3.,2.,4.,5.]));g=torch.Generator().manual_seed(9)
        counts=torch.bincount(torch.multinomial(q,20000,True,generator=g),minlength=5)/20000
        self.assertTrue(bool(((counts-q).abs()<.015).all()))

    def test_exact_pg_gradient(self):
        l=torch.tensor([.2,-.1,.6],dtype=torch.double,requires_grad=True);r=torch.tensor([-.4,.2,.8],dtype=torch.double)
        p=l.softmax(0);g=torch.autograd.grad((p*r).sum(),l,retain_graph=True)[0]
        expected=sum(p[i].detach()*r[i]*torch.autograd.grad(p.log()[i],l,retain_graph=True)[0] for i in range(3))
        self.assertTrue(torch.allclose(g,expected,atol=1e-12))
        self.assertLess(float(r[0]/.02),0.)

    def test_ppo_ratio_one(self):
        lp=torch.randn(3,4).log_softmax(1).requires_grad_();choices=torch.tensor([0,1,2])
        loss,info=ppo_loss(lp,lp.detach(),choices,torch.tensor([-.2,0.,.5]),lp.detach())
        self.assertTrue(torch.equal(info['ratio'],torch.ones(3)));loss.backward()

    def test_nearopt_fallback(self):
        self.assertEqual(near_optimal([0.,-.1,.01]).tolist(),[True,False,False])
        self.assertEqual(near_optimal([0.,.1,.098],positive_evidence=True).tolist(),[False,True,True])

    def test_root_partition_and_normalizer(self):
        DataAccessLedger.disjoint_roots({'fit':{1},'calibration':{2},'audit':{3}})
        with self.assertRaises(AssertionError):DataAccessLedger.disjoint_roots({'fit':{1},'audit':{1}})
        with self.assertRaises(AssertionError):FitNormalizer().fit(torch.randn(4,3),'audit')

    def test_class_permutation_equivariance(self):
        v=Verifier();x=torch.randn(5,14);a=torch.randn(4,10);order=torch.tensor([3,0,4,1,2])
        r,d=v(x,a);rr,dd=v(x[order],a)
        self.assertTrue(torch.allclose(r,rr,atol=1e-7));self.assertTrue(torch.allclose(d[:,order],dd,atol=1e-7))

    def test_class_action_alignment(self):
        v=Verifier();x=torch.randn(5,14);a=torch.randn(4,5,10);order=torch.tensor([3,0,4,1,2])
        r,d=v(x,a);rr,dd=v(x[order],a[:,order])
        self.assertTrue(torch.allclose(r,rr,atol=1e-7));self.assertTrue(torch.allclose(d[:,order],dd,atol=1e-7))

    def test_delayed_return_is_distinct(self):
        # Explicit synthetic delayed-only environment; not real-task evidence.
        immediate={'macro':0.,'tail':None};delayed={'macro':.3,'tail':.1}
        self.assertEqual(utility([immediate])['R'],0.)
        self.assertAlmostEqual(utility([immediate,immediate,delayed])['R'],.1)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.state,self.model,self.opt,self.counter=fixture()

    def test_candidate_legality_dedup_and_reconstruction(self):
        desc=describe(self.state,self.model,self.opt,self.counter,seed=5)
        candidates=generate(self.state,self.model,desc,5)
        self.assertLessEqual(len(candidates),16);self.assertEqual(len({c.key for c in candidates}),len(candidates))
        self.assertEqual([c.key for c in candidates],[c.key for c in generate(self.state,self.model,desc,5)])
        self.assertTrue(any(c.removals>0 for c in candidates))
        zero,bal=anchors(self.state,self.model,5)
        for c in candidates:validate_candidate(c,self.state,zero if c.anchor=='ZERO' else bal)

    def test_all_pool_and_B_lt_C(self):
        s=VisibleState(self.state.pool,None,self.state.observed,self.state.birth,0,len(self.state.pool),self.state.groups,True,self.state.access)
        desc=describe(s,self.model,self.opt,self.counter)
        self.assertEqual(len(generate(s,self.model,desc)),1)
        s=VisibleState(s.pool,None,s.observed,s.birth,0,2,s.groups,True,s.access)
        with self.assertRaises(AssertionError):s.validate(self.model)

    def test_snapshot_adam_rng_buffers(self):
        self.model.register_buffer('extra',torch.ones(3))
        s=Snapshot(self.model,self.opt);m,o,_=s.restore();a=torch.rand(3);m.extra.zero_();o.param_groups[0]['lr']=.7
        mm,oo,_=s.restore();b=torch.rand(3)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(mm.extra,torch.ones(3)))
        self.assertEqual(oo.param_groups[0]['lr'],.001);self.assertTrue(oo.state_dict()['state'])

    def test_query_future_eviction_firewall(self):
        bank=self.state.pool;groups=self.state.groups
        ledger=DataAccessLedger({i:0 for i in groups},groups,{str(i):'S' for i in groups},0)
        ledger.check(bank,'support')
        with self.assertRaises(AssertionError):ledger.check(bank,'query')
        ledger.frontier=-1
        with self.assertRaises(AssertionError):ledger.check(bank,'support')
        ledger.frontier=0
        with self.assertRaises(AssertionError):ledger.check(bank,'support',[])

    def test_selector_rejects_future_or_query_pool(self):
        self.state.access.frontier=-1
        with self.assertRaisesRegex(AssertionError,'FUTURE_IDS'):prepare(self.state,self.model,self.opt,1,self.counter)
        self.state.access.frontier=0
        self.state.access.roles[str(int(self.state.pool.parents[0]))]='AUDIT'
        with self.assertRaisesRegex(AssertionError,'QUERY_SUPPORT_FIREWALL'):prepare(self.state,self.model,self.opt,1,self.counter)

    def test_risk_counts_group_units(self):
        groups={i:str(i//2) for i in self.state.groups}
        roles={g:'S' for g in groups.values()}
        access=DataAccessLedger({i:0 for i in groups},groups,roles,0)
        s=VisibleState(self.state.pool,None,self.state.observed,self.state.birth,0,24,groups,True,access)
        desc=describe(s,self.model,self.opt,self.counter)
        self.assertEqual([d['available_units'] for d in desc['diagnostics']],[4,4,4])
        self.assertTrue(all(not d['noise']['available'] for d in desc['diagnostics']))

    def test_feature_and_candidate_inputs_exclude_query(self):
        from ..contracts import VisibleState
        self.assertNotIn('query',VisibleState.__dataclass_fields__)
        a=prepare(self.state,self.model,self.opt,8,self.counter)
        unused_query=torch.randn(20,512);unused_query.add_(100)
        b=prepare(self.state,self.model,self.opt,8,self.counter)
        self.assertTrue(torch.equal(a[1],b[1]));self.assertTrue(torch.equal(a[2],b[2]))

    def test_integer_counts_and_missing_masks(self):
        q=self.state.pool.parents_only(torch.arange(4));c=evaluate(self.model,q)
        self.assertEqual(c.denominator.tolist(),[4,0,0]);self.assertIsNone(c.ba({2}))
        self.assertEqual(paired(c,c,{0})['macro'],0.)
        self.assertEqual(utility([paired(c,c,{2})])['tail_coverage'],0.)

    def test_frozen_seal_detects_parameter_changes(self):
        v=Verifier();p=ResidualPolicy();old=fingerprint([v,p],('candidates','mask'))
        with torch.no_grad():next(v.parameters()).add_(1)
        self.assertNotEqual(old,fingerprint([v,p],('candidates','mask')))

    def test_branch_null_order_and_action_effect(self):
        pool=self.state.pool
        support_ids=torch.tensor([i for i in range(24) if i%8<6]);query_ids=torch.tensor([i for i in range(24) if i%8>=6])
        support=pool.parents_only(support_ids);query=pool.parents_only(query_ids)
        model=Learner(2);model.trained_groups=set();model.arrive(support,3);opt=optimizer(model)
        learn(model,opt,support,2,1,self.counter,None,8,'engineering');model.trained_groups={str(int(i)) for i in support_ids}
        roles={str(i):('S' if i%8<6 else 'AUDIT') for i in range(24)}
        ledger=DataAccessLedger({i:0 for i in range(24)},self.state.groups,roles,0)
        state=VisibleState(support,None,(6,)*3,(0,)*3,0,24,self.state.groups,True,ledger)
        candidates,classes,actions=prepare(state,model,opt,5,self.counter)
        env=PulseEnvironment(model,opt,state,[],[query],{0},ledger,self.counter,post_steps=2,pre_steps=2,batch=8)
        before={k:v.clone() for k,v in model.state_dict().items()}
        out=env.compare(candidates,33)
        self.assertEqual(out[0]['summary']['R'],0.)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        reversed_out=env.compare([candidates[0]]+list(reversed(candidates[1:])),33)
        self.assertEqual({x['candidate'].key:x['summary']['R'] for x in out},{x['candidate'].key:x['summary']['R'] for x in reversed_out})
        zero=env.rollout(candidates[0].memory,33)
        self.assertTrue(any(not torch.equal(v,zero['model'].state_dict()[k]) for k,v in before.items()))
        if len(candidates)>1:
            alt=env.rollout(candidates[1].memory,33)
            self.assertTrue(any(not torch.equal(v,alt['model'].state_dict()[k]) for k,v in zero['model'].state_dict().items()))

