"""Regression for near-tied sparse-class coverage; synthetic S features only."""
import importlib
import json
import os
import time
from pathlib import Path
import torch
from .coverage import OriginalChoice, CanonicalChoice
from . import trajectory as T


def run(output):
    start = time.time()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native = importlib.import_module('r112_evidence_memory.native')
    disagreements, old_failures, pools_checked, gap = 0, 0, 0, 0.
    for device in ['cpu', 'cuda']:
        for seed in range(32):
            rng = torch.Generator().manual_seed(113000 + seed)
            generic = torch.nn.functional.normalize(torch.randn(136, 6, 512, generator=rng), dim=-1).to(device)
            labels = torch.cat([torch.zeros(100, dtype=torch.long), torch.arange(1, 19).repeat_interleave(2)])
            evidence = native.fixture(generic, len(labels), 0, labels)
            S, C = evidence.role(0), evidence.C
            original = OriginalChoice(S, C, 256 - 2 * C, evidence.core_keys())
            fixed = CanonicalChoice(S, C, 256 - 2 * C, evidence.core_keys())
            pools_checked += 1
            canonical = []
            for c in range(C):
                ix = S.gidx[S.y[S.gidx] == c]
                center = torch.nn.functional.normalize(S.z[ix].mean(0), dim=0)
                required = ix[(S.z[ix] @ center).argmax()]
                canonical.append(int(required))
                old_required = ix[(S.z[ix] @ original.cent[c]).argmax()]
                disagreements += int(required != old_required)
                gap = max(gap, float((S.z[ix] @ center - S.z[ix] @ original.cent[c]).abs().max()))
            assert set(canonical) <= set(fixed.cover.tolist())
            missing = set(canonical) - set(original.cover.tolist())
            if missing and not old_failures:
                m = T.old.learner.Learner(113).to(device)
                m.arrive(S, C)
                state = T.old.visible(evidence, None, 0)
                for draw in range(4):
                    bank = original.construct('RANDOM', generator=torch.Generator(device=device).manual_seed(draw))[0]
                    try:
                        T.old.contracts.validate_candidate(T.old.contracts.Candidate(bank, 'anchor', 'E-ZERO', 0), state)
                    except AssertionError as error:
                        assert str(error) == 'HOST_COVERAGE_ANCHOR_REMOVED'
                        old_failures += 1
                        break
                for mode in ['RANDOM', 'BALANCED']:
                    bank = fixed.construct(mode, generator=torch.Generator(device=device).manual_seed(0))[0]
                    T.old.contracts.validate_candidate(T.old.contracts.Candidate(bank, 'anchor', mode, 0), state)
                    evidence.with_support(bank).validate()
            if seed >= 3 and old_failures:
                break
    assert old_failures > 0, 'REGRESSION_DID_NOT_REPRODUCE_ORIGINAL_FAILURE'
    result = {'passed': True, 'synthetic_only': True, 'real_images': 0, 'real_classifier_updates': 0,
              'pools_checked': pools_checked, 'old_centroid_coverage_disagreements': disagreements,
              'old_anchor_assertions_reproduced': old_failures, 'max_dot_product_rounding_difference': gap,
              'canonical_constructor_matches_unchanged_validator': True,
              'core_bytes_quotas_and_parent_constraints_preserved': True, 'wall_seconds': time.time() - start}
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    run(os.environ['Q_OUTPUT'])
