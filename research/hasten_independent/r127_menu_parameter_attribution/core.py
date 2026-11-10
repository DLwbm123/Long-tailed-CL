"""Fixed readout schema and additive attribution; no learned selection."""
import json
from pathlib import Path
STAGES=[list(range(5)),list(range(5,10)),list(range(10,15)),list(range(15,19)),list(range(19,23))]
ARMS=['GRPO','TARS','RAPO','TARS_RAPO','TARS_CE']
READOUTS=[('COMMON_INITIAL_MENU','COMMON',0),('COMMON_FINAL_MENU','COMMON',4)]+[(a+'_FINAL_MENU',a,4) for a in ARMS]
TEMPLATE='What is the endoscopic finding or anatomical landmark shown in this image? Choose one category from the categories seen so far: {seen_categories}. Answer with the category name only.'


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def question(stage,names):return TEMPLATE.format(seen_categories='; '.join(names[c] for cs in STAGES[:stage+1] for c in cs))


def old_access(ids,labels):
    ids=[int(i) for i in ids]
    if not ids or any(i<0 or i>=len(labels) or labels[i] not in STAGES[0] for i in ids):raise PermissionError('ONLY_ORIGINAL_FIRST_TASK_VAL_ALLOWED')
    return ids


def metrics(predictions,labels,counts):
    assert len(predictions)==len(labels)==624
    den=[labels.count(c) for c in STAGES[0]];good=[sum(p==c and y==c for p,y in zip(predictions,labels)) for c in STAGES[0]]
    recall=[g/n for g,n in zip(good,den)];head=[c for c in STAGES[0] if counts[c]>=250];assert head==[1,2,3,4]
    return {'correct':good,'denominator':den,'recall':recall,'old5_macro':sum(recall)/5,'first_task_head':sum(recall[c] for c in head)/len(head),'micro':sum(good)/624}


def decompose(a,b,c,h):
    result={'menu':b-a,'parameter_at_final_menu':c-b,'matched_total':c-a,'saved_vs_matched_readout_residual':h-c,'historical_total':h-a}
    assert abs(result['menu']+result['parameter_at_final_menu']-result['matched_total'])<1e-12
    assert abs(result['menu']+result['parameter_at_final_menu']+result['saved_vs_matched_readout_residual']-result['historical_total'])<1e-12
    return result


def fixtures():
    labels=[0,1,5];old_access([0,1],labels);denied=0
    for ids in [[2],[-1],[],[3]]:
        try:old_access(ids,labels)
        except PermissionError:denied+=1
    assert denied==4
    r=decompose(.8,.7,.4,.41);assert abs(r['menu']+.1)<1e-12 and abs(r['saved_vs_matched_readout_residual']-.01)<1e-12
    assert len(READOUTS)==7 and [s for _,_,s in READOUTS]==[0]+[4]*6
    names={i:f'CLASS{i:02d}' for i in range(23)}
    assert all((names[c] in question(0,names))==(c<5) for c in range(23))
    return {'passed':True,'additive_menu_parameter_readout_residual_identity':True,'old_VAL_access_denial_cases':4,'fixed_seven_readouts_no_menu_search':True,'new_model_calls':0}
