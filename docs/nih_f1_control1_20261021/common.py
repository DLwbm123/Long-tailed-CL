import json,os,time
from pathlib import Path
import numpy as np

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def settings():
    c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());return c,Path(c['output'])

def report(x,labels,heads,order):
    reports=[]
    for task,w in enumerate(heads,1):
        seen=order[:w.shape[1]];mask=np.isin(labels,seen);pred=np.array(seen)[(x[mask]@w.astype(np.float32)).argmax(1)]
        correct={str(k):int((pred[labels[mask]==k]==k).sum()) for k in seen}
        assert all(int((labels[mask]==k).sum())==15 for k in seen)
        reports.append(dict(task=task,seen=seen,correct=correct,validation_n=int(mask.sum()),balanced_accuracy_percent=100*sum(correct.values())/int(mask.sum())))
    final=reports[-1]['correct']
    return dict(stages=reports,final_correct=sum(final.values()),mean_BA_percent=float(np.mean([r['balanced_accuracy_percent'] for r in reports])),tail4_correct=sum(final[str(k)] for k in [7,14,15,16]),tail4_n=60,tail7_correct=sum(final[str(k)] for k in [15,16,7,14,18,12,5]),tail7_n=105,initial10_correct=sum(final[str(k)] for k in order[:10]),later10_correct=sum(final[str(k)] for k in order[10:]))
