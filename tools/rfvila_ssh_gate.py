"""Forced SSH command for the single-run backup key; no shell or forwarding."""
import os,sys
from pathlib import Path
op=os.environ.get('SSH_ORIGINAL_COMMAND','')
if op in ('status','archive','ack'):
    os.execv('/usr/bin/python3',['python3','/tmp/n63io.py',op])
elif op=='revoke':
    auth=Path.home()/'.ssh/authorized_keys';entry=Path('/tmp/n64authorized_line').read_text().strip()
    rows=auth.read_text().splitlines();assert rows.count(entry)==1
    tmp=auth.with_name('authorized_keys.n64.part');tmp.write_text('\n'.join(r for r in rows if r!=entry)+'\n');tmp.chmod(0o600);tmp.replace(auth)
    print('REVOKED')
else:raise SystemExit('COMMAND_NOT_ALLOWED')
