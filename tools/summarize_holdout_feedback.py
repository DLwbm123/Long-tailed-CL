"""Validate frozen holdout diagnostics and export anonymous, reproducible aggregates."""
import json
import math
from pathlib import Path
import sys


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def sign(value):
    return 1 if value>1e-8 else -1 if value < -1e-8 else 0


def delta(heads,bank,metric):
    return heads[bank+'_native'][metric]-heads[bank+'_static_pc'][metric]


def class_metric(head,classes,metric):
    values=[head['per_class'][str(c)] for c in classes]
    return mean(v['correct']/v['n'] if metric=='balanced_accuracy' else v['mean_square_risk'] for v in values)


def readout(root):
    root=Path(root);lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    results={};rows=[];suite_cost={};cpu={};max_residual=0.;max_bank_error=0.
    for job in lock['jobs']:
        name=job['name'];suite=json.loads((root/'suites'/name/'STATUS.json').read_text())
        status=json.loads((root/'runs'/name/'STATUS.json').read_text())
        result=json.loads((root/'runs'/name/'RESULTS.json').read_text())
        assert suite['status']==status['status']=='COMPLETE'
        assert status['completed_states']==job['stages']
        assert sorted(map(int,result['states']))==job['stages']
        assert result['dataset']==job['dataset']
        for r in (status,result):
            assert r['optimizer_updates']==r['policy_updates']==0 and r['test_accessed'] is False
        suite_cost[name]=suite['elapsed_seconds'];cpu[name]=status['cpu_process_seconds'];results[name]=result
        for task,s in result['states'].items():
            assert s['fit_meta_identity_disjoint'] and s['original_development_reproduced']
            max_bank_error=max(max_bank_error,*s['bank_reconstruction_max_error'].values())
            for solve in s['solves'].values():
                assert solve['relative_residual']<=1e-8
                max_residual=max(max_residual,solve['relative_residual'])
            assert set(s['splits'])=={'fit_current','meta_current','development_current','development_all'}
            for split in s['splits'].values():
                assert set(split['heads'])=={'all_native','all_static_pc','fit_only_native','fit_only_static_pc'}
                for h in split['heads'].values():
                    pc=h['per_class'];assert h['n']==sum(v['n'] for v in pc.values())
                    assert all(0<=v['correct']<=v['n'] and v['identity_groups']<=v['n'] for v in pc.values())
                    assert abs(h['balanced_accuracy']-mean(v['correct']/v['n'] for v in pc.values()))<1e-12
                    assert abs(h['macro_square_risk']-mean(v['mean_square_risk'] for v in pc.values()))<1e-12
            meta=s['splits']['meta_current']['heads']
            selected='native' if delta(meta,'fit_only','macro_square_risk') < -lock['feedback_risk_tolerance'] else 'static_pc'
            assert s['meta_selected_head']==selected
            row=dict(dataset=job['dataset'],task=int(task),selected=selected,contrasts={},selected_development={})
            for bank in ('all','fit_only'):
                row['contrasts'][bank]={view:{metric:delta(s['splits'][view]['heads'],bank,metric)
                    for metric in ('balanced_accuracy','macro_square_risk')} for view in s['splits']}
            for view in ('development_current','development_all'):
                heads=s['splits'][view]['heads']
                row['selected_development'][view]={metric:heads['fit_only_'+selected][metric]-heads['fit_only_static_pc'][metric]
                    for metric in ('balanced_accuracy','macro_square_risk')}
            old=[c for c in s['seen'] if c not in s['current_classes']]
            heads=s['splits']['development_all']['heads']
            row['selected_development']['development_old']={metric:None if not old else
                class_metric(heads['fit_only_'+selected],old,metric)-class_metric(heads['fit_only_static_pc'],old,metric)
                for metric in ('balanced_accuracy','macro_square_risk')}
            row['class_direction_counts']={}
            for bank in ('all','fit_only'):
                row['class_direction_counts'][bank]={}
                for metric in ('balanced_accuracy','macro_square_risk'):
                    pairs=[]
                    for c in s['current_classes']:
                        values=[]
                        for view in ('meta_current','development_current'):
                            hh=s['splits'][view]['heads'];values.append(sign(class_metric(hh[bank+'_native'],[c],metric)-class_metric(hh[bank+'_static_pc'],[c],metric)))
                        pairs.append(values)
                    row['class_direction_counts'][bank][metric]=dict(total=len(pairs),same=sum(a==b for a,b in pairs),
                        both_nonzero=sum(a!=0 and b!=0 for a,b in pairs),same_nonzero=sum(a==b and a!=0 for a,b in pairs),
                        both_tie=sum(a==b==0 for a,b in pairs))
            rows.append(row)
    assert len(rows)==lock['expected_states']==9 and len(rows)*4==lock['expected_heads']==36
    summary={}
    for dataset in ('ISIC','HK'):
        rr=[r for r in rows if r['dataset']==dataset];summary[dataset]=dict(states=len(rr),native_selected=sum(r['selected']=='native' for r in rr),directions={},selected_mean={})
        for bank in ('all','fit_only'):
            summary[dataset]['directions'][bank]={}
            for metric in ('balanced_accuracy','macro_square_risk'):
                pairs=[(sign(r['contrasts'][bank]['meta_current'][metric]),sign(r['contrasts'][bank]['development_current'][metric])) for r in rr]
                summary[dataset]['directions'][bank][metric]=dict(same=sum(a==b for a,b in pairs),total=len(pairs),
                    both_nonzero=sum(a!=0 and b!=0 for a,b in pairs),same_nonzero=sum(a==b and a!=0 for a,b in pairs),both_tie=sum(a==b==0 for a,b in pairs))
        for view in ('development_current','development_all','development_old'):
            summary[dataset]['selected_mean'][view]={metric:mean(r['selected_development'][view][metric] for r in rr if r['selected_development'][view][metric] is not None) for metric in ('balanced_accuracy','macro_square_risk')}
    budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'],suite_gpu_seconds=suite_cost,
        round_gpu_seconds=sum(suite_cost.values()),cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+sum(suite_cost.values()),
        cpu_check_seconds=lock['cpu_selfcheck_seconds'],worker_cpu_process_seconds=cpu,
        failed_or_repair_attempts=[],original_deadline=lock['original_deadline'],note='Suite residence summed once; worker CPU separately recorded, not added to GPU residence.')
    audit=dict(status='PASS',states=9,heads=36,split_head_summaries=144,optimizer_updates=0,policy_updates=0,test_accessed=False,
        maximum_solve_residual=max_residual,maximum_bank_reconstruction_error=max_bank_error,
        original_development_reproduced=True,current_fit_meta_identity_disjoint=True,
        current_meta_no_overlap_with_historical_training_identity='asserted by completed frozen runner',
        independent_confirmation=False)
    return dict(results=results,rows=rows,summary=summary,budget=budget,audit=audit)


def self_check():
    assert [sign(v) for v in [-1.,0.,1.]]==[-1,0,1]
    assert mean([1,3])==2 and mean([]) is None
    heads={'fit_only_native':{'macro_square_risk':.2},'fit_only_static_pc':{'macro_square_risk':.3}}
    assert delta(heads,'fit_only','macro_square_risk')<0
    h={'per_class':{'1':{'n':2,'correct':1,'mean_square_risk':.2},'2':{'n':4,'correct':1,'mean_square_risk':.4}}}
    assert class_metric(h,[1,2],'balanced_accuracy')==.375
    assert math.isclose(class_metric(h,[1,2],'macro_square_risk'),.3)


if __name__=='__main__':
    cfg=json.load(sys.stdin);self_check()
    data=readout(cfg['root']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    for name,key in [('RESULTS','results'),('COMPARISONS','rows'),('SUMMARY','summary'),('BUDGET_LEDGER','budget'),('AUDIT','audit')]:
        text=json.dumps(data[key],ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        assert not any(s in text for s in ['/remote-home/','/Users/','identity_component','adapter','checkpoint'])
        (out/(name+'.json')).write_text(text)
    print(json.dumps({k:v for k,v in data.items() if k not in ('results','rows')},indent=2))
