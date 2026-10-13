"""CPU-only device-boundary and premature-receipt regression; no model or data load."""
import ast
from pathlib import Path
from types import SimpleNamespace


def run():
    tree=ast.parse(Path(__file__).with_name('run_module_additions.py').read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="config.get('preflight', False)")
    start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='check' for t in n.targets))
    function=ast.FunctionDef(name='probe',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=block.body[start:],decorator_list=[])
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'preflight_receipt','exec')
    class CPU:
        def __init__(self,value):self.value=value
    class DeviceTensor:
        def __init__(self,value):self.value=value;self.transfers=0
        def detach(self):return self
        def cpu(self):self.transfers+=1;return CPU(self.value)
    def equal(a,b):
        assert isinstance(a,CPU) and isinstance(b,CPU),'Comparison skipped device normalization'
        return a.value==b.value
    for changed in (False,True):
        writes=[];statuses=[];tensor=DeviceTensor(2 if changed else 1)
        env=dict(torch=SimpleNamespace(equal=equal,cuda=SimpleNamespace(max_memory_allocated=lambda:0)),
            encoder=SimpleNamespace(dim=1536,state_dict=lambda:{'parameter':tensor}),preflight_state={'parameter':CPU(1)},
            projection_audit={},readout_check={},controller=SimpleNamespace(updates=0),task=1,epoch=1,norm=1.,
            loss=SimpleNamespace(detach=lambda:1.),audit={},activation={},solve={},x=[0],drift_weight=0.,
            lca_enabled=True,max_residual=1e-10,lca_audit={},output=Path('unused'),
            save=lambda p,v:writes.append(v),status=lambda **v:statuses.append(v))
        exec(compiled,env)
        try:env['probe']()
        except ValueError:
            assert changed and not writes and not statuses
        else:
            assert not changed and len(writes)==1 and writes[0]['parameters_unchanged']
            assert writes[0]['optimizer_updates']==0 and statuses[0]['status']=='COMPLETE'
        assert tensor.transfers==1
    # Execute the collector's actual remote receipt block with incomplete and complete fixtures.
    collector=ast.parse(Path(__file__).with_name('collect_lca_supported.py').read_text())
    code=next(n.value for n in ast.walk(collector) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'def scalar(v)' in n.value)
    remote=ast.parse(code)
    receipt=next(n for n in ast.walk(remote) if isinstance(n,ast.If) and ast.unparse(n.test)=="(folder / 'STATUS.json').exists()" and any(isinstance(t,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='complete' for a in t.targets) for t in n.body))
    begin=next(i for i,n in enumerate(receipt.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='v' for t in n.targets))
    normalize=compile(ast.fix_missing_locations(ast.Module(body=receipt.body[begin:],type_ignores=[])),'collector_receipt','exec')
    full=dict(status='PASS',optimizer_updates=0,parameters_unchanged=True,test_accessed=False,lca_supported={},max_solve_residual=1e-10)
    for f,worker,expected in [(dict(status='PASS'),'INCOMPLETE',False),(full,'INCOMPLETE',False),(dict(status='PASS'),'COMPLETE',False),(full,'COMPLETE',True)]:
        env=dict(f=f,st={'status':worker},scalar=lambda v:{k:x for k,x in v.items() if type(x) in (float,int,bool)},preflights={},name='fixture')
        exec(normalize,env);v=env['preflights']['fixture']
        assert v['receipt_complete']==expected and (v['status']=='PASS')==expected
    cycle=ast.parse(Path(__file__).with_name('run_module_cycle.py').read_text())
    gate=next(n for n in ast.walk(cycle) if isinstance(n,ast.If) and "config.get('version') == 'supported_old_margin_v1'" in ast.unparse(n.test))
    guard=compile(ast.fix_missing_locations(ast.Module(body=[gate],type_ignores=[])),'coordinator_receipt','exec')
    for receipt,accept in [(full,True),(dict(full,parameters_unchanged=False),False),(dict(full,test_accessed=True),False),({k:v for k,v in full.items() if k!='lca_supported'},False)]:
        try:exec(guard,dict(config={'version':'supported_old_margin_v1'},receipt=receipt))
        except ValueError:assert not accept
        else:assert accept
    return dict(status='PASS',gpu_work=False,medical_images_read=False,formal_updates=0,
        device_conversion_required=True,changed_state_rejected_before_receipt=True,
        success_receipt_written_once=True,premature_PASS_rejected=True,
        coordinator_state_and_test_sealing_guard=True,
        real_GPU_fix_validation=False)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
