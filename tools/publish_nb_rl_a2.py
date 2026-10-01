"""One completion-event publication; paths and repository identity stay private."""
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request


def read_public_archive(content):
    records={}
    with tarfile.open(fileobj=io.BytesIO(content),mode='r:') as tar:
        for m in tar:
            assert m.isfile() and Path(m.name).name==m.name and m.size<2*1024**2 and m.name not in records
            assert Path(m.name).suffix in ('.json','.jsonl','.csv','.md')
            text=tar.extractfile(m).read().decode()
            assert not re.search(r'/remote-home/|/data_nas/|/Users/|"(?:password|access_token|api_key|private_key|sample_id|identity_component)"\s*:',text,re.I),m.name
            records[m.name]=text
    return records


def selfcheck():
    def archive(name,text):
        b=io.BytesIO();data=text.encode()
        with tarfile.open(fileobj=b,mode='w') as t:
            member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
        return b.getvalue()
    assert read_public_archive(archive('RUN_STATUS.json','{"status":"COMPLETE"}'))
    for name,text in [('../outside.md','x'),('weights.pt','x'),('bad.json','{"sample_id":"private"}')]:
        try:read_public_archive(archive(name,text))
        except AssertionError:pass
        else:raise AssertionError('Unsafe publication archive accepted')
    return dict(status='PASS',safe_aggregate_accepted=True,path_asset_identity_boundaries_rejected=True)


def main():
    cfg=json.loads(Path(os.environ['N84_PUBLISH_CONFIG']).read_text())
    work=Path(cfg['worktree']);dest=work/cfg['public_destination'];receipt=Path(cfg['receipt'])
    env=dict(os.environ,HTTPS_PROXY=cfg['proxy'],HTTP_PROXY=cfg['proxy'],ALL_PROXY=cfg['proxy'],NO_PROXY='',no_proxy='')
    def git(*args,**kwargs):
        return subprocess.check_output(['git','-c','http.proxy='+cfg['proxy'],*args],cwd=work,env=env,**kwargs)
    assert git('branch','--show-current').decode().strip()==cfg['branch'],'Publication branch changed'
    assert git('remote','get-url','origin').decode().strip()==cfg['remote'],'Publication repository changed'
    git('diff','--quiet');git('diff','--cached','--quiet')
    script='''import io,json,tarfile,time,sys
from pathlib import Path
root=Path(json.loads(Path('/tmp/q84.json').read_text())['root']);pub=root/'public'
status=json.loads((pub/'RUN_STATUS.json').read_text())['status']
assert status in ('COMPLETE','BLOCKED','INCOMPLETE_BUDGET')
if status=='COMPLETE':
 for _ in range(15):
  if (pub/'FINAL_BACKUP_ACK.json').exists():break
  time.sleep(2)
 assert json.loads((pub/'FINAL_BACKUP_ACK.json').read_text())['status']=='PASS'
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for p in sorted(pub.iterdir()):
  if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md'):
   assert p.stat().st_size<2*1024**2
   tar.add(p,arcname=p.name,recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''
    script=script.replace('/tmp/q84.json',cfg.get('remote_root_config','/tmp/q84.json'))
    content=subprocess.check_output(['ssh','-o','BatchMode=yes',cfg['host'],'python3 -'],input=script.encode(),timeout=75)
    records=read_public_archive(content)
    status=json.loads(records['RUN_STATUS.json'])['status']
    if status=='COMPLETE':
        complete=json.loads(records['COMPLETE.json'])
        assert (complete['stage_rows'],complete['class_rows'],complete['new_steps'])==tuple(cfg.get('expected_counts',(60,300,14100)))
        for name,count in zip(('stage_metrics.csv','class_metrics.csv'),cfg.get('expected_counts',(60,300,14100))[:2]):
            assert len(list(csv.DictReader(io.StringIO(records[name]))))==count
    dest.mkdir(parents=True,exist_ok=True)
    for name,text in records.items():(dest/name).write_text(text)
    auth=work/cfg['authorization'];value=json.loads(auth.read_text());value.update(state=status,publication_status='RESULTS_RELEASE')
    auth.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    paths=[str((dest/name).relative_to(work)) for name in records]+[cfg['authorization']]
    git('add','--pathspec-from-file=-',input=('\n'.join(paths)+'\n').encode())
    git('-c','core.whitespace=cr-at-eol','diff','--cached','--check')
    git('commit','-m','Publish completed experiment results and backup receipts')
    git('push',stderr=subprocess.STDOUT)
    sha=git('rev-parse','HEAD').decode().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':cfg['proxy'],'http':cfg['proxy']}))
    os.environ.update({k:env[k] for k in ('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','NO_PROXY','no_proxy')})
    url='https://api.github.com/repos/'+cfg['repository']+'/branches/'+urllib.parse.quote(cfg['branch'],safe='')+'?delivery='+sha
    with opener.open(url,timeout=45) as response:remote=json.load(response)
    assert remote['commit']['sha']==sha,'Remote commit differs'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(status='PUBLISHED',execution_status=status,commit=sha,
        branch=cfg['branch'],anonymous_public_access=True,unix=time.time()),indent=2)+'\n')
    print('PUBLICATION_COMPLETE',sha,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        cfg=json.loads(Path(os.environ['N84_PUBLISH_CONFIG']).read_text())
        p=Path(cfg['receipt']);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(dict(status='BLOCKED',error=type(error).__name__,message=str(error),unix=time.time()),indent=2)+'\n')
        raise
