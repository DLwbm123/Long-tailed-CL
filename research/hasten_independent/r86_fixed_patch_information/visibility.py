"""Record only owned live process strings; never restart a process for naming."""
import os,subprocess,json
from pathlib import Path
def capture(root,tag):
    ps=subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(os.getpid())+','+str(os.getppid())],text=True)
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory,gpu_uuid','--format=csv,noheader'],text=True)
    own=[line for line in rows.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    assert own,'OWN_GPU_PROCESS_NOT_VISIBLE'
    gpu_rows=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    allowed={line.split(',')[1].strip() for line in gpu_rows.splitlines()}
    assert {line.split(',')[-1].strip() for line in own}==allowed,'PHYSICAL_GPU_MAPPING'
    for bad in json.loads(os.environ.get('FORBIDDEN_PROCESS_TOKENS_JSON','[]')):
        assert bad.lower() not in (ps+'\n'+'\n'.join(own)).lower()
    (Path(root)/(tag+'_visibility.private.json')).write_text(json.dumps(dict(ps=ps,nvidia_owned_rows=own),indent=2))
