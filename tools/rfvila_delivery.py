"""Bounded local backup relay and final source/aggregate delivery, no private arrays locally."""
import io,json,os,subprocess,tarfile,time,traceback
from pathlib import Path
PUBLIC=('FINAL_REPORT_ZH.md','FINAL_REPORT.json','METHOD_MATRIX.csv','stage_metrics.csv','class_metrics.csv','summary_by_parent_and_projection.csv','lambda_cv_scores.csv','lambda_selection.json','bootstrap_intervals.csv','error_decomposition.csv','candidate_coverage.csv','projection_robustness.csv','paired_differences.csv','forgetting.csv','SOLVER_PARITY.json','ENGINEERING_REPORT.json','PROTOCOL_LOCK.json','SOURCE_LOCK.json','RANDOM_MAP_LOCK.json','ACCESS_LEDGER.json','RESOURCE_REPORT.json','BACKUP_REPORT.json','completion_receipts.json','failure_receipts.json','NEXT_DECISION.json','PROCESS_RESIDENCE.json','RESOURCE_ADMISSION.json')

def main():
    cfg=json.loads(Path(os.environ['RF_DELIVERY_CONFIG']).read_text());root=Path(cfg['worktree']);status=Path(cfg['receipt'])
    def record(s,**kw):status.write_text(json.dumps({'status':s,'time':time.time(),**kw},indent=2))
    def remote(host,op):return subprocess.run(['ssh','-o','ConnectTimeout=15',host,'python3 /tmp/n63io.py '+op],capture_output=True,text=True,check=True).stdout
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    try:
        record('RUNNING');last_summary=0
        while time.time()<cfg['deadline_epoch']:
            snap=json.loads(remote('my-gpu','status'))
            if time.time()-last_summary>=3600:
                last_summary=time.time();record('RUNNING',remote=snap)
            if snap.get('request') and not snap.get('acked'):
                req=snap['request'];record('BACKUP',token=req['token'])
                p=subprocess.Popen(['ssh','my-gpu','python3 /tmp/n63io.py archive'],stdout=subprocess.PIPE)
                q=subprocess.run(['ssh','jiangsuiyang','python3 /tmp/n63io.py receive'],stdin=p.stdout,capture_output=True,text=True)
                p.stdout.close();pc=p.wait()
                if pc or q.returncode:raise RuntimeError('BACKUP_STREAM_FAILED:'+q.stderr[-500:])
                receipt=json.loads(q.stdout)
                if receipt['token']!=req['token'] or receipt['status']!='VERIFIED':raise RuntimeError('BACKUP_VERIFICATION_FAILED')
                subprocess.run(['ssh','my-gpu','python3 /tmp/n63io.py ack'],input=q.stdout,text=True,check=True)
                record('RUNNING',last_backup=req['token'])
            if snap.get('ready'):break
            if snap.get('controller_failed'):raise RuntimeError('CONTROLLER_FAILED')
            time.sleep(5)
        else:raise TimeoutError('RUN_DEADLINE')
        record('COMMITTING');dest=root/'docs'/cfg['public_dir'];dest.mkdir(exist_ok=True)
        p=subprocess.Popen(['ssh','my-gpu','python3 /tmp/n63io.py public'],stdout=subprocess.PIPE)
        with tarfile.open(fileobj=p.stdout,mode='r|') as archive:
            for m in archive:
                if m.name not in PUBLIC or not m.isfile() or m.size>20*1024*1024:raise ValueError('PUBLIC_ALLOWLIST')
                data=archive.extractfile(m).read()
                if m.name.endswith('.csv'):data=data.replace(b'\r\n',b'\n')
                # Published failures expose only bounded error codes; raw tracebacks stay private.
                (dest/m.name).write_bytes(data)
        p.stdout.close()
        if p.wait():raise RuntimeError('PUBLIC_COPY_FAILED')
        if git('branch','--show-current')!='exp/nb2-rfvila-12h':raise ValueError('BRANCH_CHANGED')
        if git('diff','--cached','--name-only'):raise ValueError('UNRELATED_STAGED_CHANGES')
        git('add',str(dest.relative_to(root)));git('diff','--cached','--check');git('commit','-m','Publish bounded experiment results')
        sha=git('rev-parse','HEAD');record('COMMIT_READY',commit=sha,report=str(dest/'FINAL_REPORT_ZH.md'),publication='Current execution package requires explicit push authorization for this round')
    except BaseException as e:record('NEEDS_ATTENTION',error=repr(e));raise

if __name__=='__main__':main()
