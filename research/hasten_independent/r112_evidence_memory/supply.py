"""Count birth supply and saved retention without encoding or training."""
import csv,json,os,random,time
from pathlib import Path
from collections import Counter


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,restval='NA');w.writeheader()
        for row in rows:w.writerow({k:'NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def birth_roles(ids,rows):
    byclass={}
    for i in ids:byclass.setdefault(int(rows[i]['original_label']),set()).add(rows[i]['identity_component'])
    out={}
    for label,groups in byclass.items():
        units=sorted(groups);random.Random(947110+label).shuffle(units)
        q=max(1,int(.2*len(units))) if len(units)>=3 else 0
        mapping={u:'FIT' if j<q else 'AUDIT' if j<2*q else 'S' for j,u in enumerate(units)}
        out.update({i:mapping[rows[i]['identity_component']] for i in ids if int(rows[i]['original_label'])==label})
    return out


def run(root,source110,source111):
    start=time.time();cpu=time.process_time();root=Path(root);source110=Path(source110);source111=Path(source111)
    rows=list(csv.DictReader((root/'inputs.private/train.csv').open()));role=birth_roles(list(range(len(rows))),rows)
    metadata=json.loads((source111/'inputs.private/SAVED_METADATA.json').read_text());assert len(rows)==len(metadata['original_labels'])==6821
    assert all(int(r['original_label'])==metadata['original_labels'][i] for i,r in enumerate(rows))
    supply=[];trace=[];comparisons=[];fail=[]
    for folder in ['a','b']:
        cfg=json.loads((source110/'receipts.private'/folder/'CONFIG.private.json').read_text());stream=cfg['setting'];label=[cfg['order'].index(int(r['original_label'])) for r in rows]
        born={c:t for t,cs in enumerate(cfg['stages']) for c in cs}
        for c,original in enumerate(cfg['order']):
            ids=[i for i in range(len(rows)) if label[i]==c];counts=Counter(role[i] for i in ids);n=len({rows[i]['identity_component'] for i in ids})
            reason='BIRTH_INSUFFICIENT' if n<3 else 'ROLE_ALLOCATION_MISSING' if any(counts[r]==0 for r in ['S','FIT','AUDIT']) else 'SUPPLY_PRESENT'
            supply.append({'stream':stream,'original_label':original,'real_class':c,'birth_stage':born[c],'n_birth_image_components':n,'verified_patient_groups':'NA','S_birth':counts['S'],'FIT_birth':counts['FIT'],'AUDIT_birth':counts['AUDIT'],'reason':reason})
            if reason!='SUPPLY_PRESENT':fail.append(supply[-1])
        first_loss={}
        for st,keys in sorted(metadata['donor_keys'][folder].items(),key=lambda x:int(x[0])):
            stage=int(st)
            for c in range(max(cfg['stages'][stage])+1):
                ck=[k for k in keys if label[k//8]==c];parents={k//8 for k in ck};counts=Counter(role[i] for i in parents)
                for rr in ['S','FIT','AUDIT']:
                    if not counts[rr]:first_loss.setdefault((c,rr),stage)
                item={'source':'R110','stream':stream,'seed':74002,'stage':stage,'snapshot':'post_real_donor_compression','real_class':c,'original_label':cfg['order'][c],
                      'retained_parents':len(parents),'global_tokens':sum(k%8==0 for k in ck),'local_tokens':sum(k%8>0 for k in ck),'S':counts['S'],'FIT':counts['FIT'],'AUDIT':counts['AUDIT'],'verified_patient_groups':'NA'}
                for rr in ['S','FIT','AUDIT']:
                    item[rr+'_global_tokens']=sum(k%8==0 and role[k//8]==rr for k in ck);item[rr+'_local_tokens']=sum(k%8>0 and role[k//8]==rr for k in ck)
                    item[rr+'_first_loss_stage']=first_loss.get((c,rr),'NA')
                item['reason']='RETENTION_LOSS' if any(counts[r]==0 for r in ['S','FIT','AUDIT']) else 'SUPPLY_PRESENT';trace.append(item)
        cc=json.loads((source111/'readiness_results'/folder/'CLASS_INPUT_COUNTS.json').read_text())
        for x in cc:
            trace.append({'source':'R111','stream':stream,'seed':x['seed'],'stage':x['stage'],'snapshot':'pre_current_compression_oldtail_only','real_class':x['real_class'],'original_label':x['original_label'],
                          'retained_parents':x['S']+x['FIT']+x['AUDIT'],'global_tokens':'NA','local_tokens':'NA','S':x['S'],'FIT':x['FIT'],'AUDIT':x['AUDIT'],'verified_patient_groups':'NA',
                          'S_first_loss_stage':'NA','FIT_first_loss_stage':'NA','AUDIT_first_loss_stage':'NA','reason':'RETENTION_LOSS' if x['missing_roles'] else 'SUPPLY_PRESENT',
                          'trace_limit':'R111 did not persist token identities or all intermediate donor histories; earliest loss step and view counts are unavailable'})
        for x in json.loads((source111/'readiness_results'/folder/'ROOT_INPUT_CHECKS.json').read_text()):
            comparisons.append({'source':'R111','stream':stream,'seed':x['seed'],'stage':x['stage'],'scenario':x['scenario'],'root_S_tokens':x.get('root_tokens','NA'),'capacity':256,
                                'reason':'NO_LEGAL_ACTION' if x['reason']=='no_pressure_or_action' else 'RETENTION_LOSS' if x['reason']=='missing_support_or_query' else x['reason'],
                                'prefix_loss':'NA when prefix construction was skipped; protected classes persist in constructed roots','eligible':x['eligible']})
    table(root/'SOURCE_ROLE_SUPPLY.csv',supply);table(root/'ROLE_RETENTION_TRACE.csv',trace);table(root/'SAVED_CONSTRUCTION_COMPARISON.csv',comparisons)
    result={'passed':not fail,'classes_per_flow':23,'flows':2,'minimum_birth_components':min(x['n_birth_image_components'] for x in supply),'missing_birth_role_classes':len(fail),
            'verified_patient_groups':'NA','no_role_reassignment':True,'original_role_seed':947110,'new_encoder_rows':0,'new_classifier_forward':0,'new_optimizer':0,'cost_CPU_seconds':time.process_time()-cpu,'cost_wall_seconds':time.time()-start}
    write(root/'SOURCE_SUPPLY_RESULT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run(os.environ['Q_SOURCE'],os.environ['Q_R110'],os.environ['Q_R111'])
