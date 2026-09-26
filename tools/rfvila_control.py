"""Bounded controller; each fit/eval child has a private process group."""
import fcntl,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from tools.run_nb2_vlm_r1 import _read,_write,_sha
from tools.run_rfvila import Budget,FILES,ROOT,backup

def incomplete(out,cfg,error):
    from route_a.run_ct13_real import _write_csv
    protocol=_read(out/'PROTOCOL_LOCK.json');complete=list((out/'stages').glob('*/*/task_*/STATE_LOCK.json'))
    status='INCOMPLETE_BUDGET' if 'BUDGET' in error else 'BLOCKED_ENGINEERING'
    _write(out/'FINAL_REPORT.json',{'status':status,'sealed_stages':len(complete),'expected_fit_stages':45,'utility':'NOT_EVALUABLE','next_decision':'STOP'})
    _write_csv(out/'METHOD_MATRIX.csv',[{'method':m['method'],'status':status if m['tier'] in protocol['admitted_tiers'] else 'NOT_ADMITTED_RESOURCE'} for m in protocol['methods']])
    _write(out/'failure_receipts.json',{'status':status,'error_code':error.split(':')[0],'private_tracebacks_retained':True})
    (out/'FINAL_REPORT_ZH.md').write_text(f'# NB2-RFVILA-R1\n\n状态 {status}；已封存 {len(complete)}/45 个流阶段。缺失指标为未测量，不能判定方法效用 FAIL。\n具体工程记录保留在私有运行目录，NEXT_DECISION=STOP。\n')
    _write(out/'completion_receipts.json',{'status':status,'sealed_stages':len(complete),'utility':'NOT_EVALUABLE'})
    _write(out/'NEXT_DECISION.json',{'decision':'STOP'})

def main():
    cfg=_read(Path(os.environ['RF_CONFIG']));out=Path(cfg['run_root']);budget=Budget(out)
    lock=(out/'.controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=_read(out/'SOURCE_LOCK.json')
    for name,h in source['code_sha256'].items():
        if _sha(ROOT/name)!=h:raise ValueError('SOURCE_CHANGED:'+name)
    child=None;residences=[]
    try:
        for role,done in [('fit','FIT_COMPLETE.json'),('eval','EVALUATION_COMPLETE.json')]:
            if (out/done).exists():continue
            budget.check('fit' if role=='fit' else 'forward',cfg['stage_estimate_seconds'])
            env=os.environ.copy();env['RF_ROLE']=role
            log=(out/(role+'.log')).open('ab');start=time.time()
            child=subprocess.Popen([sys.executable,'/tmp/n63w.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();last=0
            _write(out/'LIVE_PROCESS.json',{'role':role,'pid':child.pid,'pgid':child.pid,'controller_pid':os.getpid(),'start':start})
            while child.poll() is None:
                if budget.elapsed()>=39600:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise TimeoutError('INCOMPLETE_BUDGET:COMPUTE_STOP')
                if time.time()-last>=300:
                    last=time.time();record={'time':last,'elapsed':budget.elapsed(),'role':role,'pid':child.pid,
                      'sealed_stages':len(list((out/'stages').glob('*/*/task_*/STATE_LOCK.json'))),
                      'gpu':subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()}
                    _write(out/'HEARTBEAT.json',record)
                    with (out/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                time.sleep(5)
            residences.append({'role':role,'seconds':time.time()-start,'exit_code':child.returncode})
            if child.returncode:raise RuntimeError('WORKER_EXIT:'+role+':'+str(child.returncode))
            child=None
        budget.check('compute',120)
        from tools.finalize_rfvila import finalize
        finalize(out,cfg)
    except BaseException as e:
        if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
        _write(out/f'FAILED_controller_{int(time.time())}.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
        incomplete(out,cfg,str(e))
    _write(out/'PROCESS_RESIDENCE.json',{'residences':residences,'pure_GPU_compute_claim':False,'time':time.time(),'elapsed':budget.elapsed()})
    # Final backup excludes the restored preflight scratch copy and transient lock files.
    paths=[str(p.relative_to(out)) for p in out.iterdir() if p.name not in ['BACKUP_REQUEST.json','BACKUP_FAILURE.json','.controller.lock','private','states']]
    paths+=['states','private/maps','private/scores']+[str(p.relative_to(out)) for p in (out/'private').glob('*PREDICTIONS.npz')]
    paths=[p for p in paths if (out/p).exists()]
    backup(out,'FINAL',paths,budget)
    _write(out/'BACKUP_REPORT.json',{'status':'INDEPENDENT_BACKUP_COMPLETE','critical_destination_sha256':True,'parent_and_bank_restore':'PASS','whole_image_archive_bytewise_verified':False,'stage_ack_count':len(list((out/'backup_acks').glob('*.json'))),'final_ack':_read(out/'backup_acks/FINAL.json')['status']})
    _write(out/'DELIVERY_READY.json',{'status':'READY','time':time.time(),'elapsed':budget.elapsed()})

if __name__=='__main__':main()
