import unittest,torch,json
from ..base.memory import from_views
from ..base.learner import Learner,optimizer
from ..base.contracts import VisibleState,DataAccessLedger,token_keys,validate_candidate
from ..base.state_features import describe
from ..base.anchor_candidates import anchors
from ..candidates import structural,matched_random,overlap_profile,generate
from ..episodes import Roles,root,score_runs
from ..common import Counts


class PilotTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(110);n=36
        bank=from_views(torch.nn.functional.normalize(torch.randn(n,6,512),dim=-1),torch.arange(n),torch.arange(n)//12)
        groups={i:str(i) for i in range(n)};access=DataAccessLedger({i:0 for i in groups},groups,{str(i):'S' for i in groups},0)
        state=VisibleState(bank,None,(12,)*3,(0,)*3,0,48,groups,False,access)
        model=Learner(110);model.arrive(bank,3);opt=optimizer(model);desc=describe(state,model,opt,{})
        return state,model,opt,desc

    def test_structural_quotas_and_parent_coverage(self):
        state,m,o,desc=self.fixture();zero,bal=anchors(state,m,110)
        for anchor in [zero,bal]:
            memory=structural(state,anchor,desc)
            from ..base.contracts import Candidate
            validate_candidate(Candidate(memory,'structural','ZERO',0),state,anchor)

    def test_matched_edit_scale(self):
        state,m,o,desc=self.fixture();zero,bal=anchors(state,m,110)
        for anchor in [zero,bal]:
            structure=structural(state,anchor,desc);random,attempts=matched_random(state,anchor,structure,110)
            self.assertIsNotNone(random);self.assertEqual(overlap_profile(random,anchor),overlap_profile(structure,anchor))

    def test_pool_bound_unique_and_zero(self):
        s,m,o,d=self.fixture();c,a=generate(s,m,d,110)
        self.assertLessEqual(len(c),16);self.assertEqual(len(c),len({x.key for x in c}));self.assertEqual(c[0].anchor,'ZERO')
        self.assertEqual([x.key for x in c],[x.key for x in generate(s,m,d,110)[0]])

    def test_counter_phase_costs(self):
        c=Counts();c['paired_classifier_updates']=1;c['engineering_virtual_classifier_optimizer']=2
        c.phase='science';c['paired_classifier_updates']=3;c['engineering_virtual_classifier_optimizer']=4
        self.assertEqual(dict(c),{'native_paired_classifier_updates':1,'native_virtual_classifier_optimizer':2,'science_paired_classifier_updates':3,'science_virtual_classifier_optimizer':4})

    def test_complete_budget(self):
        self.assertEqual(32*(1400+2*16*700)+3000+32,764632)

    def test_roles_order_invariant_and_immutable(self):
        rows=[{'original_label':str(i//10),'identity_component':str(i)} for i in range(30)]
        a=Roles(947110);b=Roles(947110)
        a.arrive(list(range(30)),rows)
        for c in [2,0,1]:b.arrive(list(range(c*10,c*10+10)),rows)
        self.assertEqual(a.role,b.role)
        with self.assertRaises(AssertionError):a.arrive([0],rows)

    def test_actual_root_nested_query_and_H3_null(self):
        torch.manual_seed(110);rows=[{'original_label':str(i//10),'identity_component':str(i)} for i in range(60)]
        bank=from_views(torch.nn.functional.normalize(torch.randn(60,6,512),dim=-1),torch.arange(60),torch.arange(60)//10)
        prev=bank.parents_only(torch.arange(30));current=bank.parents_only(torch.arange(30,60))
        roles=Roles(947110);roles.arrive(list(range(30)),rows);roles.arrive(list(range(30,60)),rows)
        cfg={'stages':[[0,1,2],[3,4,5]],'train_counts':[10]*6};counter=Counts();counter.phase='science'
        built,skip=root(prev,current,roles,rows,cfg,1,0,counter);self.assertIsNone(skip)
        env,m,o,state,subsets,meta=built
        for h in range(3):
            for role in ['FIT','AUDIT']:self.assertTrue(subsets[h][role+'_small']<=subsets[h][role+'_large'])
        zero=anchors(state,m,110)[0];run=env.rollout(zero,31001)
        scores=score_runs([run],env,subsets)
        for name,values in scores.items():self.assertEqual(values[0]['H3']['R'],0.)
        self.assertEqual(counter['science_paired_classifier_updates'],700)

if __name__=='__main__':unittest.main()

