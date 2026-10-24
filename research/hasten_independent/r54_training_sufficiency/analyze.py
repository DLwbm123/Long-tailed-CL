"""Run once after campaign exit0/all phases complete; no model re-evaluation."""
import os,json,subprocess,statistics
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);assert json.loads((r/'STATUS.json').read_text())['phase']=='COMPLETE';out=r/'results';out.mkdir(exist_ok=True);audit=[]
for ds in ['NIH','ISIC','HK']:
 for mode in ['native','flat','factorized']:
  for fold in [0,1]:
   d=r/'audit'/mode/ds/str(fold);rec=json.loads((d/'RESULTS.json').read_text());assert rec['fit_holdout_disjoint'] and rec['fit_holdout_uncapped'] and rec['epochs']==10 and rec['VAL_reads']==rec['TEST_reads']==0;curves=[json.loads(z) for z in (d/'curves/74002_stage0.jsonl').read_text().splitlines()];assert [v['epoch'] for v in curves]==list(range(11)) and all(v['observation_state_rng_restored'] for v in curves)
   vals=[z['objective_mean'] for z in curves[-3:]];plateau=(max(vals)-min(vals))/max(abs(statistics.mean(vals)),1e-12)<.01
   audit.append(dict(dataset=ds,mode=mode,fold=fold,record=rec,curves=curves,last3_objective_plateau_1percent=plateau,not_convergence_proof=True))
reports={}
for mode in ['native','flat','factorized']:
 m=r/'full'/mode;env=os.environ.copy();env['RUN_ROOT']=str(m);subprocess.run([os.environ.get('ANALYSIS_PYTHON','/tmp/q108v/bin/python'),str(Path(__file__).with_name('analyze_mode.py'))],env=env,check=True)
 reports[mode]=json.loads((m/'results/ANALYSIS.json').read_text());assert json.loads((m/'results/VERIFICATION.json').read_text())['new_optimizer_steps']==44070
 for ds in ['NIH','ISIC','HK']:
  rr=json.loads((m/ds/'aggregate.json').read_text())['results']
  for z in rr:
   cc=[json.loads(t) for t in (m/ds/f"curves/{z['seed']}_stage{z['stage']}.jsonl").read_text().splitlines()];assert [v['epoch'] for v in cc]==list(range(11)) and all(v['observation_state_rng_restored'] for v in cc)
summary={ds:{mode:reports[mode][ds]['summary']['final'] for mode in reports} for ds in reports['factorized']}
(out/'TRAINING_AUDIT_CURVES.json').write_text(json.dumps(audit,indent=2));(out/'THREE_ARM_SUMMARY.json').write_text(json.dumps(summary,indent=2));(out/'VERIFICATION.json').write_text(json.dumps(dict(audit_models=18,full_stages=135,full_optimizer_steps=132210,all_curves_present=True,no_interim_VAL_selection=True,one_structure_only_not_general_screen_accuracy=True),indent=2))
print(json.dumps(summary,indent=2))
