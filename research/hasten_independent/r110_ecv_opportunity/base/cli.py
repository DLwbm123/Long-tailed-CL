"""Config-driven neutral entry; explicit engineering admission only."""
import json
import os
import time
from pathlib import Path
import torch
from .budget_ledger import BudgetLedger,require_admission


def main():
    cfg=json.loads(Path(os.environ['Q_CONFIG']).read_text())
    require_admission(cfg);torch.set_num_threads(min(2,cfg.get('threads',1)));torch.manual_seed(cfg.get('seed',109))
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    import subprocess,getpass
    argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()
    assert not any(s in argv.lower() for s in [getpass.getuser().lower(),'long-tailed','hasten','ecv_set'])
    (out/'ARGV.private.json').write_text(json.dumps({'pid':os.getpid(),'argv':argv}))
    ledger=BudgetLedger(out/'COST_LEDGER.jsonl',cfg['deadline_epoch'])
    mode=cfg['command'];ledger.append('start',command=mode,science=False)
    if cfg.get('dry_run'):
        result=dict(dry_run=True,command=mode,implemented=mode in {'audit','smoke','collect-pairs','train-verifier','train-sft','train-pg','evaluate','report','real-integration'})
        assert result['implemented']
    elif mode=='smoke':
        from .smoke import run
        result=run(out)
    elif mode=='collect-pairs':
        from .smoke import synthetic_environment,record_pairs
        env,m,o,s,counter=synthetic_environment(cfg.get('seed',109))
        record,candidates,outcomes=record_pairs(env,m,o,s,counter,cfg.get('seed',109))
        torch.save(record,out/'PAIRS.private.pt');result=dict(synthetic_only=True,candidates=len(candidates),counts=counter)
    elif mode in {'train-verifier','train-sft'}:
        from .verifier import Verifier
        from .policy import ResidualPolicy
        from .train_controller import train_verifier,train_sft
        rows=torch.load(cfg['records'],map_location='cpu',weights_only=False)
        records=rows if isinstance(rows,list) else [rows]
        assert len(records)*cfg.get('epochs',2)<=cfg['max_updates']
        v=Verifier(cfg.get('dynamics',True))
        if mode=='train-verifier':result=train_verifier(v,records,epochs=cfg.get('epochs',2));torch.save(v.state_dict(),out/'VERIFIER.private.pt')
        else:
            v.load_state_dict(torch.load(cfg['verifier'],weights_only=True));p=ResidualPolicy()
            result=train_sft(p,v,records,epochs=cfg.get('epochs',2));torch.save(p.state_dict(),out/'SFT.private.pt')
    elif mode=='train-pg':
        from .policy import ResidualPolicy
        from .verifier import Verifier
        from .train_controller import train_pg
        batch=torch.load(cfg['trajectories'],weights_only=False,map_location='cpu')
        p=ResidualPolicy();ref=ResidualPolicy();v=Verifier()
        p.load_state_dict(torch.load(cfg['policy'],weights_only=True));ref.load_state_dict(torch.load(cfg['reference'],weights_only=True))
        v.load_state_dict(torch.load(cfg['verifier'],weights_only=True));v.eval().requires_grad_(False);ref.eval().requires_grad_(False)
        result=train_pg(p,ref,v,batch,cfg['sigma_fit'],epochs=cfg.get('epochs',2))
        torch.save(p.state_dict(),out/'PG.private.pt')
    elif mode=='real-integration':
        from .real_integration import run
        result=run(cfg)
    elif mode=='evaluate':
        from .verifier import Verifier
        from .policy import ResidualPolicy,verifier_prior
        record=torch.load(cfg['records'],weights_only=False,map_location='cpu')
        v=Verifier();v.load_state_dict(torch.load(cfg['verifier'],weights_only=True));v.eval().requires_grad_(False)
        p=ResidualPolicy();p.load_state_dict(torch.load(cfg['policy'],weights_only=True));p.eval().requires_grad_(False)
        with torch.no_grad():
            values,_=v(record['classes'],record['actions']);col=record['return_column'];q=verifier_prior(values[:,col])
            prob=p.log_probs(record['classes'],record['actions'],q).exp();index=int(torch.multinomial(prob,1,generator=torch.Generator().manual_seed(cfg.get('seed',109))))
        assert bool(record['target_mask'][index,col])
        result=dict(saved_feedback_evaluation_only=True,split=record['split'],action=index,real_return=float(record['targets'][index,col]),regret=float(record['targets'][:,col].max()-record['targets'][index,col]))
    elif mode in {'audit','report'}:
        contract=Path(__file__).parent/'CONTRACT_LOCK.json'
        result=json.loads(contract.read_text()) if contract.exists() else dict(contract='not yet frozen',science_admitted=False)
    else:raise ValueError('UNKNOWN_COMMAND')
    (out/(mode.upper().replace('-','_')+'_RESULT.json')).write_text(json.dumps(result,indent=2))
    ledger.append('exit',command=mode,result=result,science=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

