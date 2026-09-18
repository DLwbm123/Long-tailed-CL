"""Exercise the actual service ACK command while a reader races the writer."""
import ast,json,subprocess,tempfile,threading
from pathlib import Path
source=ast.parse((Path(__file__).resolve().parents[1]/'tools/serve_ct3p.py').read_text())
command=next(n.value for n in ast.walk(source) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('cat > '))
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'ACK.json';p.write_text('{"id": -1}')
    command=command.replace('/tmp/p22root/output/private',d)
    errors=[];stop=threading.Event()
    def reader():
        while not stop.is_set():
            try: assert isinstance(json.loads(p.read_text())['id'],int)
            except BaseException as e: errors.append(repr(e));break
    t=threading.Thread(target=reader);t.start()
    try:
        for i in range(50):
            proc=subprocess.Popen(['sh','-c',command],stdin=subprocess.PIPE)
            payload=json.dumps(dict(id=i,padding='x'*65536)).encode()
            proc.stdin.write(payload[:100]);proc.stdin.flush()
            proc.stdin.write(payload[100:]);proc.stdin.close();assert proc.wait()==0
    finally: stop.set();t.join()
    assert not errors,errors
    assert json.loads(p.read_text())['id']==49
print('PASS: concurrent ACK reads saw only complete JSON')
