"""Check paired randomness separately from the unchanged class-order schedule."""
import torch
import run_nb_rl_a1 as r
from preflight_nb_rl_a2 import check as existing_check


def check(run):
    assert set(r.RUN_IDS) == {199301,199302,199401,199402,199501,199502}
    for order in (1993,1994,1995):
        a,b=order*100+1,order*100+2
        assert r.RUN_ORDERS[a] == r.RUN_ORDERS[b] == r.ISIC_ORDERS[order]
        assert (r.training_seed(a),r.training_seed(b)) == (73001,73002)
        assert not torch.equal(r.loader_generator(a,1,1).get_state(),r.loader_generator(b,1,1).get_state())
    assert torch.equal(r.loader_generator(199301,1,1).get_state(),r.loader_generator(199401,1,1).get_state())
    existing_check(run)
