"""One independent resource guard, one original clock, one native attempt."""
import fcntl
import importlib
import json
import os
import resource
import signal
import subprocess
import time
import traceback
from pathlib import Path

common = importlib.import_module('r110_ecv_opportunity.common')
rss_tree = importlib.import_module('r110_ecv_opportunity.controller').rss_tree


def summarize(root):
    rows, blocks = [], []
    for name in ['a', 'b']:
        data = json.loads((root / name / 'RESULTS.json').read_text())
        rows.extend(data['stages'])
        for seed in [74003, 74004]:
            zero = {r['stage']: r for r in data['stages'] if r['seed'] == seed and r['arm'] == 'ZERO'}
            for arm in ['ZERO', 'BAL', 'MYOPIC_GRPO', 'LONG_GRPO', 'SUP_SHARED']:
                rr = sorted([r for r in data['stages'] if r['seed'] == seed and r['arm'] == arm], key=lambda r: r['stage'])
                macro = [r['macro_BA'] - zero[r['stage']]['macro_BA'] for r in rr]
                tail = [r['oldtail_BA'] - zero[r['stage']]['oldtail_BA'] for r in rr if r['oldtail_BA'] is not None]
                score = sum(macro) / len(macro) - 2 * max([max(0, -d - .005) for d in tail], default=0)
                blocks.append({'stream': data['stream'], 'seed': seed, 'arm': arm, 'R_real5': score,
                               'mean_macro_BA': sum(r['macro_BA'] for r in rr) / 5,
                               'terminal_macro_BA': rr[-1]['macro_BA'],
                               'macro_delta': sum(macro) / 5, 'worst_oldtail_delta': min(tail) if tail else None,
                               'nonzero_learned_actions': sum(r['action'] != 0 for r in rr)})
    comparisons = []
    for stream in ['ordered', 'stratified_shuffled']:
        for seed in [74003, 74004]:
            bb = {b['arm']: b for b in blocks if b['stream'] == stream and b['seed'] == seed}
            comparisons.append({'stream': stream, 'seed': seed,
                                'LONG_minus_controls_R': {a: bb['LONG_GRPO']['R_real5'] - bb[a]['R_real5']
                                                          for a in ['ZERO', 'BAL', 'MYOPIC_GRPO', 'SUP_SHARED']},
                                'LONG_oldtail_protected': bb['LONG_GRPO']['worst_oldtail_delta'] is None or bb['LONG_GRPO']['worst_oldtail_delta'] >= -.005})
    signal_ok = all(c['LONG_oldtail_protected'] and all(v > 0 for v in c['LONG_minus_controls_R'].values()) for c in comparisons)
    common.write(root / 'FINAL_RESULTS.json', {'decision': 'development_signal_only' if signal_ok else 'no_consistent_long_GRPO_advantage',
                                             'blocks': blocks, 'comparisons': comparisons,
                                             'independent_population_confirmation': False, 'VAL_TEST': 0})


def run(cfg):
    root = Path(cfg['root_output'])
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    children, exits = {}, {}
    reason, term, peak, disk_peak, last_heartbeat = None, None, 0, 0, 0

    def limits(cores):
        os.sched_setaffinity(0, set(cores))
        resource.setrlimit(resource.RLIMIT_CPU, (400000, 400000))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2**30, 2**30))
    try:
        available = int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1]) * 1024
        assert available >= 64 * 2**30
        for n in ['a', 'b']:
            w = json.loads((root / n / 'CONFIG.private.json').read_text())
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(w['physical_gpu']), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 2**20
            assert free >= 14 * 2**30
            env = {**os.environ, **w['environment'], 'Q_SOURCE': str(root.resolve()), 'Q_MODULE': 'worker',
                   'Q_CONFIG': str(root / n / 'CONFIG.private.json')}
            with (root / n / 'worker.private.log').open('w') as log:
                children[n] = subprocess.Popen([cfg['python'], str(root / 'main.py')], env=env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                                preexec_fn=lambda c=w['cpu_affinity']: limits(c))
        common.write(root / 'LAUNCH.private.json', {'controller': os.getpid(), 'workers': {n: p.pid for n, p in children.items()},
                                                   'start': cfg['start'], 'deadline': cfg['deadline']})
        while len(exits) < 2:
            for n, p in children.items():
                if n in exits:
                    continue
                pid, status, usage = os.wait4(p.pid, os.WNOHANG)
                if pid:
                    p.returncode = os.waitstatus_to_exitcode(status)
                    exits[n] = {'exit_code': p.returncode, 'CPU_seconds': usage.ru_utime + usage.ru_stime}
                    if p.returncode:
                        reason = reason or 'WORKER_FAILURE'
            if not (root / 'ADMISSION.json').exists() and not (root / 'ADMISSION_FAILED.json').exists():
                paths = [root / n / 'NATIVE_ADMISSION.json' for n in ['a', 'b']]
                if all(p.exists() for p in paths):
                    receipts = [json.loads(p.read_text()) for p in paths]
                    passed = all(r['passed'] and r['complete_science_projected_seconds'] < cfg['deadline'] - time.time() for r in receipts)
                    common.write(root / ('ADMISSION.json' if passed else 'ADMISSION_FAILED.json'),
                                 {'passed': passed, 'workers': receipts, 'epoch': time.time()})
            rss = rss_tree(os.getpid())
            peak = max(peak, rss)
            allocated = 0
            for p in root.rglob('*'):
                try:
                    if p.is_file() and not p.is_symlink():
                        s = p.stat()
                        allocated += max(s.st_size, s.st_blocks * 512)
                except FileNotFoundError:
                    pass
            disk_peak = max(disk_peak, allocated)
            if time.time() >= cfg['deadline'] - 120:
                reason = reason or 'ORIGINAL_CLOCK'
            if rss >= 29_000_000_000 or any(rss_tree(p.pid) >= 28_000_000_000 for n, p in children.items() if n not in exits):
                reason = reason or 'RAM_LIMIT'
            if allocated + 512 * 2**20 >= 20_000_000_000:
                reason = reason or 'DISK_LIMIT'
            status = {'remaining_seconds': cfg['deadline'] - time.time(), 'peak_RSS_bytes': peak,
                      'allocated_bytes': allocated, 'reason': reason, 'exits': exits}
            common.write(root / 'GUARD.private.json', status)
            if time.time() - last_heartbeat >= 1800:
                common.write(root / 'HEARTBEAT.private.json', {'epoch': time.time(), **status})
                last_heartbeat = time.time()
            if reason:
                sig = signal.SIGTERM if term is None else signal.SIGKILL if time.time() - term > 10 else None
                if sig:
                    for n, p in children.items():
                        if n not in exits:
                            try:
                                os.killpg(p.pid, sig)
                            except ProcessLookupError:
                                pass
                    term = time.time() if term is None else term
            time.sleep(2)
        if reason:
            raise RuntimeError(reason)
        if (root / 'ADMISSION_FAILED.json').exists():
            common.write(root / 'STATUS.json', {'phase': 'COMPLETE_NOT_RESOURCE_ADMITTED'})
        else:
            summarize(root)
            common.write(root / 'STATUS.json', {'phase': 'COMPLETE_READY_FOR_DELIVERY'})
    except BaseException:
        reason = reason or 'CONTROL_FAILURE'
        (root / 'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc())
        for n, p in children.items():
            if n not in exits:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                    os.waitpid(p.pid, 0)
                except ProcessLookupError:
                    pass
        common.write(root / 'STATUS.json', {'phase': 'FAILED_PRESERVED', 'reason': reason})
    finally:
        common.write(root / 'CONTROL_EXIT.json', {'exit_code': int(reason is not None), 'reason': reason,
                                                 'workers': exits, 'original_elapsed_seconds': time.time() - cfg['start'],
                                                 'peak_group_RSS_bytes': peak, 'peak_disk_bytes': disk_peak})


if __name__ == '__main__':
    run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
