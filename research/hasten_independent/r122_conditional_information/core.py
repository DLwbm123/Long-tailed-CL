import json
import math
from pathlib import Path

METHODS=['ROOT','CENTER','GLOBAL_REPEAT','CENTER_DETAIL']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def parse_box(text):
    try:
        text=text.strip()
        if text.startswith('```'):
            lines=text.splitlines();assert len(lines)>=3 and lines[-1].strip()=='```';text='\n'.join(lines[1:-1])
        box=json.loads(text)
        if isinstance(box,dict):assert set(box)=={'bbox'};box=box['bbox']
        assert isinstance(box,list) and len(box)==4
        assert all(type(x) is int for x in box)
        assert all(0<=x<=1000 for x in box)
        assert box[2]-box[0]>=100 and box[3]-box[1]>=100
        return box,True
    except (ValueError,TypeError,AssertionError):return [0,0,1000,1000],False


def crop(image,box):
    w,h=image.size;x1,y1,x2,y2=box
    return image.crop((math.floor(w*x1/1000),math.floor(h*y1/1000),math.ceil(w*x2/1000),math.ceil(h*y2/1000)))


def fixtures():
    from PIL import Image
    for text in ['[100,100,900,900]','{"bbox":[100,100,900,900]}','```json\n[100,100,900,900]\n```']:
        assert parse_box(text)==([100,100,900,900],True)
    for text in ['no box','[true,0,1000,1000]','[0.0,0,1000,1000]','[-1,0,1000,1000]','[0,0,50,1000]','[900,0,100,1000]']:
        assert parse_box(text)==([0,0,1000,1000],False)
    assert crop(Image.new('RGB',(100,80)),[250,250,750,750]).size==(50,40)
    return {'passed':True,'real_images':0,'valid_formats':3,'invalid_formats':6}


def score_rows(rows,counts):
    den=[sum(r['class']==c for r in rows) for c in range(23)];assert all(n>0 for n in den)
    results=[]
    for method in METHODS:
        correct=[sum(r['class']==c and r['results'][method]['recognition']['prediction']==c for r in rows) for c in range(23)]
        recall=[c/n for c,n in zip(correct,den)]
        ids={'macro':list(range(23)),'head':[c for c,n in enumerate(counts) if n>=250],
             'tail':[c for c,n in enumerate(counts) if n<250],'tiny':[c for c,n in enumerate(counts) if n<25],
             'non_tiny':[c for c,n in enumerate(counts) if n>=25]}
        result={'name':method,'correct':correct,'denominator':den,'micro':sum(correct)/sum(den)}
        result.update({k:sum(recall[c] for c in group)/len(group) for k,group in ids.items()})
        if method in METHODS[2:]:
            props=[r['results'][method]['proposal'] for r in rows];recs=[r['results'][method]['recognition'] for r in rows]
            result.update(valid_fraction=sum(p['valid'] for p in props)/len(props),
                          vision_encoder_rows=sum(p['vision_encoder_rows']+r['vision_encoder_rows'] for p,r in zip(props,recs)),
                          input_tokens=sum(p['input_tokens']+r['input_tokens'] for p,r in zip(props,recs)),
                          generated_tokens=sum(p['generated_tokens']+r['generated_tokens'] for p,r in zip(props,recs)),
                          elapsed_seconds=sum(p['seconds']+r['seconds'] for p,r in zip(props,recs)))
        results.append(result)
    return results


def summarize(rows,counts):
    results=score_rows(rows,counts);parts={part:score_rows([r for r in rows if r['split']==part],counts) for part in ['A','B']}
    baseline,main=results[2:]
    gates={'macro_increment_ge_1pp':main['macro']-baseline['macro']>=.01,'tail_increment_ge_2pp':main['tail']-baseline['tail']>=.02,
           'head_increment_ge_minus1pp':main['head']-baseline['head']>=-.01,'non_tiny_increment_positive':main['non_tiny']>baseline['non_tiny'],
           'A_non_tiny_increment_positive':parts['A'][3]['non_tiny']>parts['A'][2]['non_tiny'],
           'B_non_tiny_increment_positive':parts['B'][3]['non_tiny']>parts['B'][2]['non_tiny'],
           'GLOBAL_valid_fraction_ge_90pct':baseline['valid_fraction']>=.9,'CENTER_valid_fraction_ge_90pct':main['valid_fraction']>=.9,
           'equal_visual_and_input_token_costs':main['vision_encoder_rows']==baseline['vision_encoder_rows'] and main['input_tokens']==baseline['input_tokens'],
           'CENTER_generated_tokens_not_larger':main['generated_tokens']<=baseline['generated_tokens']}
    assert len(gates)==10
    return {'scope':'frozen_instrument_TRAIN_only_information_intervention_not_RL_training','results':results,'partitions':parts,
            'gates':gates,'all_gates_pass':all(gates.values()),'TRAIN_components':len(rows),'VAL':0,'TEST':0,
            'new_optimizer_updates':0,'classifier_in_sample_to_TRAIN':True,'independent_patient_validation':False,
            'different_proposed_boxes':sum(r['results']['GLOBAL_REPEAT']['proposal']['box']!=r['results']['CENTER_DETAIL']['proposal']['box'] for r in rows)}
