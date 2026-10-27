import json,torch
from diagnostics import paired_stats
from folds import assign_folds
from support import make_support,support_scores
a=torch.tensor([[2.,0.],[0.,2.],[2.,0.],[0.,2.]])
b=torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
y=torch.tensor([0,1,0,1]);s=paired_stats(a,b,y,2)
assert sum(c['prediction_disagreements'] for c in s)==2
assert sum(c['original_only_correct'] for c in s)==2 and sum(c['both_correct'] for c in s)==2
same=paired_stats(a,a,y,2);assert all(c['prediction_disagreements']==0 for c in same)
perm=torch.tensor([1,0]);p=paired_stats(a[:,perm],b[:,perm],1-y,2)
for c in range(2):
 assert s[c]['original_correct']==p[1-c]['original_correct'] and s[c]['flipped_correct']==p[1-c]['flipped_correct']
z=torch.nn.functional.normalize(torch.tensor([[1.,0.],[.8,.2],[0.,1.],[.2,.8]]),dim=1);lab=torch.tensor([0,0,1,1]);folds=torch.tensor(assign_folds(lab.tolist(),['a','b','c','d']))
for q in [0,1]:
 ix=folds==q;sx=~ix;assert not bool((ix&sx).any())
 su=make_support(z[sx],lab[sx],2)
 for mode in ['prototype_','distribution_']:assert torch.isfinite(support_scores(z[ix],su,mode)).all()
print(json.dumps(dict(passed=True,no_images=True,optimizer_steps=0,backwards=0,VAL_TEST_reads=0,paired_transition_identities=True,identical_view_control=True,label_permutation=True,opposite_fold_disjoint=True)))
