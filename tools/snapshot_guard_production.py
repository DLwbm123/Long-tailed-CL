"""Fixed-frame native head guard; no dataset, optimizer, or GPU launcher."""
import time
import torch
import core1_competition as competition
import module_additions as modules
import native_pointwise_guard_design as native
import snapshot_full_guard_design as dual

VERSION='snapshot_guard_v1'


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
    return 0


def system(bank,pairs,R):
    k=len(bank['n']);mass=R.new_full((k,),1/k)
    return native.system(bank['Q']/k,bank['mu']/k,mass,pairs,R,R.new_zeros((len(R),k)),proximal=0.)


@torch.no_grad()
def reference_head(bank,local,module,dataset,seen):
    initial,R,_=modules.head(bank,module,dataset,seen,local)
    return competition.bank_head(bank,competition.competition(bank,initial),regularizer=R)[0]


class Session:
    def __init__(self,config):
        self.enabled=config.get('snapshot_guard_mode')=='on';self.version=0;self.events=[];self.current=None
        self.invalidations=0;self.keys=set();self.config=config;self.supports={}
    def register(self,task,fit_ids,all_ids):
        if task in self.supports or len(fit_ids)==0 or len(fit_ids.unique())!=len(fit_ids) or not torch.equal(all_ids,torch.arange(len(all_ids),device=all_ids.device)) or not torch.isin(fit_ids,all_ids).all():
            raise ValueError('Guard fit registration differs')
        self.supports[task]=(cpu(fit_ids).clone(),cpu(all_ids).clone())
    def capture(self,steps):
        if steps!=self.version:raise ValueError('Encoder/support version differs')
        return steps
    def require(self,certificate,steps):
        if self.current is not certificate or steps!=self.version or certificate['snapshot']['encoder_version']!=str(steps):
            raise ValueError('Snapshot guard certificate expired')
    def advance(self,steps):
        if steps!=self.version+1:raise ValueError('Encoder step skipped/repeated')
        self.invalidations+=self.current is not None;self.current=None;self.version=steps
    @torch.no_grad()
    def project(self,head,bank,pairs,R,off_bank,off_local,x,y,row_ids,all_row_ids,old_count,
                event,task,epoch,steps,support_version):
        if not self.enabled:return head,None
        if steps!=self.version or support_version!=steps:raise ValueError('Features are stale after encoder update')
        if (row_ids.ndim!=1 or len(row_ids)!=len(x) or len(row_ids.unique())!=len(row_ids)
            or all_row_ids.ndim!=1 or not torch.isin(row_ids,all_row_ids).all()):raise ValueError('Guard fit row identity differs')
        if task not in self.supports or not torch.equal(cpu(row_ids),self.supports[task][0]) or not torch.equal(cpu(all_row_ids),self.supports[task][1]):
            raise ValueError('Full original fit scope required, no meta/subset substitution')
        key=(event,task,epoch,steps)
        if key in self.keys:raise ValueError('Guard snapshot applied twice')
        frame=f'task_{task}_epoch_{epoch}' if event=='epoch_init' else f'task_{task}_sealed'
        if bank.get('_frame')!=frame:raise ValueError('Guard bank frame differs')
        snap=dict(event=event,scope='fit',encoder_eval=True,frame=frame,support_frame=frame,
                  encoder_version=str(steps),support_encoder_version=str(support_version))
        dual.validate_snapshot(snap)
        start=time.monotonic();threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            b,off,loc,rr,aa,xx,yy=map(cpu,(bank,off_bank,off_local,R,pairs,x,y))
            reference=reference_head(off,loc,self.config['module'],self.config['name'],self.config['order'][:len(b['n'])])
            op,rhs,inv=system(b,aa,rr);wu,base_audit=inv(rhs)
            if not torch.allclose(wu,head.detach().cpu(),atol=2e-8,rtol=2e-8):raise ValueError('Native snapshot head differs')
            w,a,cert=dual.solve(op,rhs,inv,wu,reference,xx,yy,old_count,snap)
            if a['status']!='CERTIFIED':raise ValueError('Full-fit guard unresolved; no identity fallback')
            cert.update(version=VERSION,proposal_head=wu,task=task,epoch=epoch,steps=steps,
                seen=self.config['order'][:len(b['n'])],source_commit=self.config['source_commit'],
                module=self.config['module'],authoritative_dtype='float64',support_n=len(xx),
                independent_feasibility_recheck_without_support=False)
            a.update(event=event,task=task,epoch=epoch,steps=steps,protected_fit_n=len(xx),
                support_version=support_version,head_dtype='float64',extra_image_extractions=0,
                boundary_force_payload_bytes=tensor_bytes({k:cert[k] for k in ('force','reference','head','proposal_head')}),
                logical_host_input_tensor_bytes=sum(tensor_bytes(v) for v in (bank,off_bank,off_local,R,pairs,x,y)) if head.device.type!='cpu' else 0,
                native_rebuild_residual=base_audit['relative_residual'],operation_wall_seconds=time.monotonic()-start,
                constraints_detached=True,encoder_version_invalid_after_next_update=True)
            self.keys.add(key);self.current=cert;self.events.append(a)
            return w.to(device=head.device),cert
        finally:torch.set_num_threads(threads)


def verify_checkpoint(st,config,task,steps):
    from run_snapshot_guard_scope import metadata,STEPS,SIZES
    if not 1<=task<=len(SIZES[config['name']]) or steps!=sum(STEPS[config['name']][:task]) or st['seen']!=config['order'][:sum(SIZES[config['name']][:task])]:
        raise ValueError('Guard stage order/budget differs')
    if st.get('guard_metadata')!=metadata(config,task,steps):raise ValueError('Guard checkpoint metadata differs')
    cert=st.get('snapshot_guard_state')
    if (not cert or cert.get('version')!=VERSION or cert.get('steps')!=steps or cert.get('task')!=task
        or cert.get('source_commit')!=config['source_commit'] or cert.get('seen')!=st['seen']
        or cert.get('module')!=config['module'] or cert.get('authoritative_dtype')!='float64'
        or cert['snapshot']['event']!='boundary_final' or cert['snapshot']['encoder_version']!=str(steps)
        or st['bank'].get('_frame')!=cert['snapshot']['frame'] or st.get('pmt_frame')!=cert['snapshot']['frame']):
        raise ValueError('Wrong/stale guarded checkpoint')
    from paired_transport_design import VERSION as frame_version
    if st['bank'].get('_transport_version')!=frame_version:raise ValueError('Guard transport schema differs')
    if cert['head'].dtype!=torch.float64 or not torch.equal(cert['head'].float(),st['head']):raise ValueError('Guard float compatibility head differs')
    if not torch.equal(cert['head'],st['pmt_head_double']):raise ValueError('Guard authoritative head differs')
    R,_=modules.metric(st['bank'],config['module'],config['name'],st['seen'],st['module_memory'])
    op,rhs,inv=system(st['bank'],st['boundary_pair_weights'].double(),R)
    residual=dual.verify_stationarity(op,rhs,cert,cert['snapshot'])
    rebuilt,a=inv(rhs+cert['force'])
    error=float((rebuilt-cert['head']).norm()/(1+cert['head'].norm()))
    c=cert['certificate']
    if (error>1e-8 or c['status']!='CERTIFIED' or c['minimum_raw_slack'] < -2e-8
        or c['stationarity']>1e-8 or c['complementarity']>1e-7
        or abs(c['primal_dual_gap'])>1e-7*(1+abs(c['objective_candidate']))
        or c['protection_event']!='boundary_final' or cert['support_n']*(len(st['seen'])-1)!=c['constraint_count']):
        raise ValueError('Guard numerical certificate differs')
    return cert['head'],dict(rebuild_relative_error=error,stationarity=residual,independent_full_support_rechecked=False)


def scores(features,state,config,task,steps):
    head,audit=verify_checkpoint(state,config,task,steps)
    return features.double()@head.to(features.device),audit
