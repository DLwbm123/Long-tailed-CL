import json,torch
from check_decision import check
from diagnostic_stats import check as stats_check
from folds import assign_folds
x=check();x.update(stats_check());labels=[0,0,0,1,1,1,1];components=['a','b','c','d','e','f','g'];f=assign_folds(labels,components);assert set(f)=={0,1};x.update(metadata_split=True,controller_optimizer_steps=0);print(json.dumps(x))
