"""One background execution pipeline: remote job, backup, aggregate publication.

This waits on the job's SSH exit once; it does not poll or schedule monitoring.
Paths and the expected branch are provided in a private local config.
"""
import io,json,os,subprocess,tarfile,time,urllib.request
from pathlib import Path

PUBLIC=('FINAL_REPORT_ZH.md','FINAL_REPORT.json','METHOD_MATRIX.csv','stage_metrics.csv',
 'class_metrics.csv','summary_by_seed.csv','bootstrap_intervals.csv','forgetting.csv',
 'error_decomposition.csv','paired_differences.csv','medical_vlm_audit.json','exposure_audit.md',
 'resource_report.json','access_ledger.json','backup_report.json','protocol_lock.json',
 'source_lock.json','completion_receipts.json','NEXT_DECISION.json','QUALIFICATION.json',
 'GENERIC_PRIOR_AUDIT.csv','GENERIC_PRIOR_AUDIT.json','GENERIC_REPRODUCTION.json')

def main():
    cfg=json.loads(Path(os.environ['MED_DELIVERY_CONFIG']).read_text())
    receipt=Path(cfg['receipt']);work=Path(cfg['worktree'])
    def record(status,**kw):
        receipt.write_text(json.dumps({'status':status,'time':time.time(),**kw},indent=2)+'\n')
    def run(args,**kw):return subprocess.run(args,check=True,**kw)
    def git(*args,**kw):
        return run(['git','-c','http.proxy=http://127.0.0.1:7897','-c','http.version=HTTP/1.1',*args],cwd=work,capture_output=True,text=True,**kw).stdout.strip()
    try:
        record('RUNNING')
        run(['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','my-gpu','/tmp/m62l.sh'])
        record('EXPERIMENT_FINISHED_BACKUP')
        backup={'status':'SECOND_BACKUP_UNVERIFIED','primary_preserved':True}
        try:
            run(['ssh','my-gpu','python3 /tmp/m62io.py inventory'])
            producer=subprocess.Popen(['ssh','my-gpu','python3 /tmp/m62io.py archive'],stdout=subprocess.PIPE)
            consumer=subprocess.run(['ssh','jiangsuiyang','/tmp/m62r.sh'],stdin=producer.stdout)
            producer.stdout.close();code=producer.wait()
            if code or consumer.returncode:raise RuntimeError('backup copy failed')
            check=json.loads(run(['ssh','jiangsuiyang','python3 /tmp/m62verify.py'],capture_output=True,text=True).stdout)
            if check['status']!='PRIMARY_MATRIX_COMPLETE':raise RuntimeError('backup completion receipt missing')
            backup={'status':'INDEPENDENT_NFS_COPY_COMPLETE','primary_preserved':True,
                'critical_assets':'COPIED_BEFORE_RUN','new_run':'COPIED_AFTER_COMPLETION',
                'source_sha256_inventory':True,'copy_exit_code':0,'destination_completion_receipt':check,
                'verification':'successful streaming copy and readable completion receipt; no full destination rehash',
                'destination_bytewise_verification':False}
        except Exception as e:backup['error']=str(e)
        run(['ssh','my-gpu','python3 /tmp/m62io.py backup_status'],input=json.dumps(backup),text=True)
        result=subprocess.Popen(['ssh','my-gpu','python3 /tmp/m62io.py public'],stdout=subprocess.PIPE)
        dest=work/cfg['public_dir'];dest.mkdir(parents=True,exist_ok=True)
        received=set()
        with tarfile.open(fileobj=result.stdout,mode='r|') as archive:
            for member in archive:
                if member.name not in PUBLIC or not member.isfile() or member.size>20*1024*1024:raise ValueError('UNEXPECTED_PUBLIC_MEMBER')
                (dest/member.name).write_bytes(archive.extractfile(member).read());received.add(member.name)
        result.stdout.close()
        if result.wait() or received!=set(PUBLIC):raise RuntimeError('INCOMPLETE_PUBLIC_DELIVERY')
        report=json.loads((dest/'FINAL_REPORT.json').read_text());complete=json.loads((dest/'completion_receipts.json').read_text())
        if report['stage_rows']!=complete['stage_rows']:raise ValueError('REPORT_COVERAGE')
        for p in dest.glob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
        if git('branch','--show-current')!=cfg['branch']:raise ValueError('BRANCH_CHANGED')
        # Never accidentally include another task's staged changes.
        if git('diff','--cached','--name-only'):raise ValueError('UNRELATED_STAGED_CHANGES')
        git('fetch','origin')
        git('merge','--no-edit','@{u}')
        names='\n'.join(str((dest/name).relative_to(work)) for name in PUBLIC)+'\n'
        git('add','--pathspec-from-file=-',input=names)
        git('diff','--cached','--check')
        git('commit','-m','Publish completed medical encoder matrix and bounded results')
        git('push','origin','HEAD')
        sha=git('rev-parse','HEAD')
        refs=dict(line.split()[::-1] for line in git('ls-remote','origin').splitlines())
        if refs.get('refs/heads/'+cfg['branch'])!=sha:raise ValueError('REMOTE_SHA_MISMATCH')
        url='https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/'+sha+'/'+cfg['public_dir']+'/FINAL_REPORT_ZH.md'
        op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
        with op.open(url,timeout=30) as response:
            if response.status!=200:raise ValueError('ANONYMOUS_ACCESS_FAILED')
        record('DELIVERED',commit=sha,branch=cfg['branch'],primary_gate=report['primary_utility_gate'],backup_status=backup['status'],anonymous_http=200)
    except BaseException as e:
        record('NEEDS_ATTENTION',error=repr(e));raise

if __name__=='__main__':main()
