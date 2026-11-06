"""Maximum-size native simulation and formal memory profiles, original-clock admission."""
import time,copy,json,subprocess,os,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from memory import from_views,BUDGET_BYTES
from manager import ValueNet,Policy,Choice
from learner import Learner,optimizer,learn,frozen
from meta import Meta
from common import guard,write
ARMS=['BALANCED', 'VALUE_GREEDY', 'VALUE_GREEDY_MORE', 'ZERO_ACTOR', 'REWARD_SHUFFLED', 'GSPO_SET', 'GSPO_SET_LT']

def run(c,encoder,rows,y,first_ids,counter):
    r=Path(c['output']);selected=[]
    for cl in range(5):
        ids=[i for i in first_ids if int(y[i])==cl];groups=set()
        for i in ids:
            if rows[i]['identity_component'] not in groups:selected.append(i);groups.add(rows[i]['identity_component'])
            if len(groups)==6:break
    assert len(selected)==30,'NATIVE_NEEDS_30_DISTINCT_FIRST_STAGE_COMPONENTS'
    times=[]
    for _ in range(2):
        torch.cuda.synchronize();st=time.perf_counter();z=encoder.encode(c,rows,selected,counter,'native',first_ids);torch.cuda.synchronize();times.append((time.perf_counter()-st)/30)
    # Repeat only in engineering fixture to profile max support/counterfactual/policy shapes.
    zz=torch.stack([z[(j//24)*6+j%6] for j in range(120)]);fake=from_views(zz,torch.arange(120,device='cuda'),torch.arange(120,device='cuda')//24);fake_rows=[{'identity_component':'fixture'+str(i)} for i in range(120)]
    torch.manual_seed(c['seed']);value=ValueNet().cuda();more=copy.deepcopy(value);initial=Policy().cuda();actors={name:copy.deepcopy(initial) for name in ['ZERO_ACTOR','REWARD_SHUFFLED','GSPO_SET','GSPO_SET_LT']};meta=Meta(c,value,more,actors);e=meta.episode(fake,fake_rows,0,8,True);legacy=meta.episode(fake,fake_rows,0,8,False);assert e['B']==64 and e['C']==[2,3,5]
    groups=[]
    for _ in range(2):
        torch.cuda.synchronize();start=time.perf_counter();out=meta.group(e,counter,'native',legacy);torch.cuda.synchronize();groups.append(time.perf_counter()-start)
    # Largest legal arrival parent pool for formal selectors, synthetic labels/features only.
    N=c['maximum_stage_parent_pool'];features=z[torch.arange(N,device='cuda')%30];pool=from_views(features,torch.arange(N,device='cuda'),torch.arange(N,device='cuda')%23);m=Learner(c['seed']).cuda();m.arrive(pool,23);o=optimizer(m);teacher=frozen(m);step=[]
    for _ in range(2):
        torch.cuda.synchronize();st=time.perf_counter();learn(m,o,pool,4,c['seed'],counter,teacher,batch=64,kind='native');torch.cuda.synchronize();step.append((time.perf_counter()-st)/4)
    for role in ['REWARD_SHUFFLED','GSPO_SET','GSPO_SET_LT']:
        assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in actors[role].parameters()),'NO_POLICY_GRADIENT'
    assert torch.count_nonzero(actors['ZERO_ACTOR'].net[-1].weight)==0
    assert not any(v.requires_grad or v.grad is not None for v in encoder.visual.parameters())
    choices=[];profiles={}
    for arm in ARMS:
        mode='BALANCED' if arm=='BALANCED' else 'VALUE' if arm.startswith('VALUE_') else 'POLICY';results=[]
        for C in [5,23]:
            profpool=from_views(features,torch.arange(N,device='cuda'),torch.arange(N,device='cuda')%C);profmodel=Learner(c['seed']).cuda();profmodel.arrive(profpool,C)
            for _ in range(2):
                torch.cuda.synchronize();st=time.perf_counter();ch=Choice(profpool,profmodel,[N//C+1]*C,[0]*C,0,256,value);mem,_,info=ch.construct(mode,actor=actors.get(arm),generator=torch.Generator(device='cuda').manual_seed(c['seed']),record=False);torch.cuda.synchronize();dt=time.perf_counter()-st
            results.append(dt)
        choices.append(max(results));profiles[arm]={'second_pass_seconds_max_C5_C23':max(results),'bytes':len(mem)*2064+184,'global_tokens':info['global_tokens']}
    # Includes full current encoding, one shared sealed VAL pass, all meta and formal work.
    estimated=groups[1]*16*5+sum(choices)*5+step[1]*7*5*300+times[1]*(6821+1698)+600
    projection=2*estimated+7200
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert any(v.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in v for v in gpu.splitlines());passed=projection<c['deadline']-time.time() and torch.cuda.max_memory_reserved()<12*2**30
    write(r/'BUDGET.json',{'passed':passed,'own_GPU_verified':True,'native_distinct_TRAIN_images':30,'native_encoder_passes':2,'max_shadow_complete_group_seconds_second':groups[1],'max_formal_classifier_step_seconds_second':step[1],'max_formal_pool_parents':N,'formal_memory_profiles':profiles,'sixview_encoder_seconds_per_image_second':times[1],'per_stream_meta_groups_max':80,'per_stream_shadow_classifier_updates_max':51840,'per_stream_formal_classifier_updates':10500,'per_stream_policy_updates_max':480,'per_stream_value_updates_max':2240,'safety':2,'IO_analysis_reserve_seconds':7200,'projection_per_stream_seconds':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory_budget_bytes':BUDGET_BYTES,'synthetic_rows_components_labels_only_engineering':True,'all_fixture_parameters_discarded':True,'native_counts':dict(counter),'VAL_TEST_images':0})
    del meta,m,o,value,more,actors,initial,fake,pool,features,mem,out;gc.collect();torch.cuda.empty_cache();guard(c);return passed
