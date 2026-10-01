"""Reuse paired analysis with a distinctly named intermediate-rate candidate."""
from report_nb_rl_a3 import report as paired_report, synthetic_check as paired_check


def report(entries,root):
    return paired_report(entries,root,candidate='M',experiment='NB-RL-A4',candidate_lr=2e-4)


def synthetic_check(root):
    return paired_check(root,candidate='M',experiment='NB-RL-A4',candidate_lr=2e-4)
