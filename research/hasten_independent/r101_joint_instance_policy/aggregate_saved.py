"""Aggregate saved training diagnostics; no model/data forward or optimizer."""
import json,pathlib,statistics,time

def run(root):
    root=pathlib.Path(root);start=time.time();cpu=time.process_time();groups={}
    for line in (root/'TRAIN_DIAGNOSTICS.private.jsonl').open():
        x=json.loads(line);groups.setdefault((x['stage'],x['arm']),[]).append(x)
    out=[]
    for (s,a),v in groups.items():
        assert len(v)==512
        row={'stage':s,'arm':a,'cycles':len(v),'supervised_grad_norm_mean':statistics.mean(x['supervised_grad_norm_before_clip'] for x in v),'supervised_clipped_fraction':statistics.mean(x['supervised_grad_norm_before_clip']>1 for x in v),'policy_grad_norm_mean':statistics.mean(x['policy_grad_norm_before_clip'] for x in v),'policy_clipped_fraction':statistics.mean(x['policy_grad_norm_before_clip']>1 for x in v)}
        for k in sorted(set().union(*(x['auxiliary'] for x in v))):row[k+'_mean']=statistics.mean(x['auxiliary'][k] for x in v)
        out.append(row)
    assert len(out)==24
    result={'records':out,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'saved_only':True,'new_optimizer_encoder_F_VAL_TEST':0}
    (root/'TRAIN_AGGREGATES.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run(pathlib.Path(__file__).resolve().parent)
