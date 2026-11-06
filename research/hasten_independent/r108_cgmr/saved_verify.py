"""Independent saved-only pilot reconstruction. No model, feature, data, forward or optimizer."""
import json,time,math
from pathlib import Path

def verify(root):
 r=Path(root);cpu=time.process_time();wall=time.time();worst=0.;total=0;gates={};same_roles={}
 for stream in ['a','b']:
  p=r/stream;rows=json.loads((p/'pilot/ROOTS.private.json').read_text());assert len(rows)==24;valid=[];old_count=0;nontrivial=0
  for x in rows:
   if 'candidate_count' not in x:continue
   n=x['candidate_count'];assert 1<=n<=9;keys=x['candidate_keys'];assert len(keys)==n;assert len({tuple(k) for k in keys})==n;base=set(keys[0]);anchors=set(x['anchor_keys']);B=len(base);assert B==x['actual_B'];assert len(anchors)==x['C0'] and all(k%8==0 for k in anchors);targets=[dict(type='NOOP',target_operations=0)]+[a for a in x['proposal_audit'] if a['present']];assert len(targets)==n
   for k,a in zip(keys,targets):
    s=set(k);assert len(s)==len(k)==B and anchors<=s;assert sum(v%8==0 for v in s)==sum(v%8==0 for v in base);assert all(v-v%8 in s for v in s);removed=base-s;added=s-base;assert len(removed)==len(added)<=min(24,B//10)
    if a['type']=='LOCAL':assert all(v%8!=0 for v in removed|added) and len(removed)==a['target_operations']
    if a['type']=='BUNDLE':
     rp=[v for v in removed if v%8==0];ap=[v for v in added if v%8==0];assert len(rp)==len(ap)==a['target_operations'];assert all(not any(v//8==g//8 for v in s) for g in rp);assert sorted(sum(v//8==g//8 for v in removed) for g in rp)==sorted(sum(v//8==g//8 for v in added) for g in ap)
   counts=x['coverage']['actual_support_counts'][:x['C0']];threshold=sorted(counts)[math.floor((len(counts)-1)/3)];T={str(i) for i,v in enumerate(counts) if v<=threshold};raw=x['raw_query_counts'];recomputed={}
   for role in ['FIT','AUDIT']:
    R=[];U=[];H=[];oldvalid=0
    for ci,branch in enumerate(raw):
     uu=[];hh=[]
     for h in range(2):
      cs=branch[role][h]['classes'];cs0=raw[0][role][h]['classes'];assert set(cs)==set(cs0)
      for c in cs:assert cs[c]['count']==cs0[c]['count'] and 0<=cs[c]['correct']<=cs[c]['count'] and 0<=cs[c]['q']<=1
      u=.5*sum(v['correct']/v['count'] for v in cs.values())/len(cs)+.5*sum(v['q'] for v in cs.values())/len(cs);worst=max(worst,abs(u-branch[role][h]['U']));uu.append(u);indices=T&set(cs)
      if indices:hh.append(sum(max(0.,cs0[c]['q']-cs[c]['q']) for c in indices)/len(indices))
     q=sum(uu)/2;harm=sum(hh)/len(hh) if hh else 0.;R.append(q-.25*harm);U.append(q);H.append(harm)
     if ci==0:oldvalid=len(hh)
    for name,a in [('R',R),('U',U),('H',H)]:
     for actual,expected in zip(a,x['scores'][role][name]):worst=max(worst,abs(actual-expected))
    recomputed[role]=dict(R=R,U=U,H=H,old=bool(oldvalid))
   chosen=max(range(n),key=lambda j:recomputed['FIT']['R'][j]);assert chosen==x['chosen_fit'];a=recomputed['AUDIT'];delta=a['R'][chosen]-a['R'][0];du=a['U'][chosen]-a['U'][0];h=a['H'][chosen];worst=max(worst,abs(delta-x['audit_delta_R']),abs(du-x['audit_delta_U']),abs(h-x['audit_H_selected']));valid.append((delta,du,h));nontrivial+=n>1;old_count+=a['old'];total+=n
  means=[sum(x[j] for x in valid)/len(valid) for j in range(3)] if valid else [None]*3;passed=nontrivial>=16 and old_count>=8 and bool(valid) and means[0]>=.005 and means[1]>=0 and means[2]<=.005;g=json.loads((p/'PILOT_GATE.json').read_text());assert passed==g['passed'];assert len(valid)==g['evaluable_roots'] and nontrivial==g['nontrivial_evaluable_roots'] and old_count==g['old_scarce_evaluable_roots']
  for a,b in zip(means,[g['mean_audit_delta_R'],g['mean_audit_delta_U'],g['mean_audit_H_selected']]):
   if a is not None:worst=max(worst,abs(a-b))
  costs=json.loads((p/'COSTS.json').read_text())['counts'];assert costs.get('A_classifier_optimizer',0)==64*len(valid)+36*sum(x['candidate_count'] for x in rows if 'candidate_count' in x);gates[stream]=dict(passed=passed,evaluable=len(valid),nontrivial=nontrivial,old_scarce=old_count,means=means);same_roles[stream]=json.loads((p/'ROLES.private.json').read_text())
 shared=set(same_roles['a'])&set(same_roles['b']);assert all(same_roles['a'][k]==same_roles['b'][k] for k in shared);assert worst<1e-12
 out={'passed':True,'scope':'independent saved pilot scores, no-op harm, candidate byte/anchor/parent counts, full gate reconstruction and observed cross-flow role consistency','all_candidate_memories_checked':total,'score_and_gate_max_error':worst,'shared_components_same_roles_count':len(shared),'streams':gates,'new_forward_optimizer':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'no_new_TEST_VAL':True};(r/'SAVED_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(verify(Path(__file__).resolve().parent))
