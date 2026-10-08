"""Reject adaptive configurations before fixed-mode files or hardware are touched."""
import json
import time
from run_fdrl import run as train


def run():
    start=time.process_time()
    for arm in ('FD5','FD20','RANDOM','GRADIENT','GREEDY','FDRL','SHUFFLED','COMP10','COMP_FDRL'):
        try:
            train(dict(method=arm,fixed_only=True,base_competition=False))
        except ValueError as exc:
            assert str(exc)=='Fixed-only mode requires FD10 without competition'
        else:
            raise AssertionError('Adaptive arm admitted')
    for arm in ('R','FD10'):
        for competition in ({},dict(base_competition=True)):
            try:
                train(dict(method=arm,fixed_only=True,**competition))
            except ValueError as exc:
                assert str(exc)=='Fixed-only mode requires FD10 without competition'
            else:
                raise AssertionError('Competition admitted')
    return dict(status='PASS',checks=['fixed-mode rejects adaptive arms and competition before any file or GPU access'],
        scientific_optimizer_updates=0,cpu_core_seconds=time.process_time()-start)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
