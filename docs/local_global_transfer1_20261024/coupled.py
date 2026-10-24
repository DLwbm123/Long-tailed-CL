"""One coupled objective derivative followed by deterministic chunk replay."""
import torch


def backward_coupled(upper, inputs, objective, record, microbatch=8):
    assert len(inputs) > 1 and not upper.training
    parameters = list(upper.parameters())
    reference = parameters[0]
    outputs = []
    rows_first = rows_replay = 0
    try:
        with torch.no_grad():
            for start in range(0, len(inputs), microbatch):
                chunk = inputs[start:start+microbatch].to(reference)
                outputs.append(upper(chunk).detach())
                rows_first += len(chunk)
        leaf = torch.cat(outputs).requires_grad_(True)
        del outputs
        loss = objective(leaf)
        record(dict(check='coupled_objective_finite', value=float(loss.detach()), PASS=bool(torch.isfinite(loss))))
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert leaf.grad is not None
        error = 0.
        for start in range(0, len(inputs), microbatch):
            chunk = inputs[start:start+microbatch].to(reference)
            actual = upper(chunk)
            error = max(error, float((actual.detach()-leaf.detach()[start:start+len(chunk)]).abs().max()))
            actual.backward(leaf.grad[start:start+len(chunk)])
            rows_replay += len(chunk)
        record(dict(check='same_canonical_replay_output', max_error=error, PASS=error <= 1e-10))
        assert error <= 1e-10
        return float(loss.detach())
    finally:
        record(dict(check='coupled_completed_rows', first_forward=rows_first, replay_forward=rows_replay,
                    replay_backward=rows_replay, scope='Completed backward rows; partial failed operator remains unknown'))
