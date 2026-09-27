"""NAS-local relay with reconnects; only one process owns the backup writer."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path

def main():
    os.umask(0o077);cfg=json.loads(Path(os.environ['RF_RELAY_CONFIG']).read_text());out=Path(cfg['root'])
    lock=(out/'.relay.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-T','-p','20048','-i',cfg['key'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','UserKnownHostsFile='+cfg['known_hosts'],'root@10.12.208.231']
    def remote(op,**kw):return subprocess.run(ssh+[op],check=True,capture_output=True,text=True,**kw).stdout
    def status(state,**kw):
        p=out/'SERVER_RELAY_STATUS.json';tmp=p.with_suffix('.part');tmp.write_text(json.dumps({'state':state,'time':time.time(),**kw}));tmp.replace(p)
    attempt=0
    while True:
        try:
            snap=json.loads(remote('status'))
            if snap.get('request') and not snap.get('acked'):
                req=snap['request'];status('COPYING',token=req['token'])
                # An existing verified receipt lets acknowledgement delivery be retried without another large copy.
                ack=out/'backup_acks'/f"{req['token']}.json"
                if ack.exists():receipt=json.loads(ack.read_text())
                else:
                    producer=subprocess.Popen(ssh+['archive'],stdout=subprocess.PIPE)
                    try:
                        consumer=subprocess.run([sys.executable,'/tmp/n63io.py','receive'],stdin=producer.stdout,capture_output=True,text=True)
                    finally:producer.stdout.close()
                    code=producer.wait()
                    if code or consumer.returncode:raise RuntimeError('BACKUP_TRANSFER_FAILED')
                    receipt=json.loads(consumer.stdout)
                if receipt['status']!='VERIFIED' or receipt['token']!=req['token']:raise ValueError('ACK_MISMATCH')
                remote('ack',input=json.dumps(receipt));status('ACKNOWLEDGED',token=req['token'])
            if snap.get('ready'):
                status('COMPLETE');remote('revoke')
                for p in [Path(cfg['key']),Path(cfg['key']+'.pub')]:p.unlink(missing_ok=True)
                return
            attempt=0;time.sleep(5)
        except (subprocess.SubprocessError,OSError,ValueError,RuntimeError) as e:
            attempt+=1;status('RECONNECTING',attempt=attempt,error=type(e).__name__)
            time.sleep(min(300,15*attempt))

if __name__=='__main__':main()
