# OPSA implementation contract

Paper-derived implementation of [OPSA](https://openreview.net/forum?id=msP4WP1nA2), an ICLR 2027 submission. This is not the authors' released implementation or a completed medical-data reproduction. The author repository inspected on 2026-10-21 does not publish the adapter, MAPC optimizer, or training harness.

## Implemented forward

`OPSAAdapter(width, rank=16)` accepts tokens of shape `[..., width]` and returns the residual-injected stream with the same shape. Place it at a chosen internal transformer residual site. No backbone or classifier is replaced automatically.

The implementation follows equations 2–5, 8–9: token normalization to radius sqrt(width); bias-free down projection; adjacent-pair quadrature `J=[[0,-1],[1,0]]`; coupled cosine gate; up projection with exact Frobenius radius gamma; residual addition. Defaults are rank 16, kappa 1, gamma 8 and MAPC delta .25. Down initialization uses std `.1/sqrt(width*rank)`. The paper does not uniquely specify Vup initialization: this implementation explicitly chooses normal std `1/sqrt(width*rank)`. There are exactly `2*width*rank` trainable parameters. Initialization is not identity.

Odd rank, zero-norm tokens, nonfinite input, and zero/nonfinite Vup norm fail rather than silently changing the equations. The module does not clip activations to the phase basin or guarantee minority-class safety.

## MAPC: explicit local interpretation

The paper mixes activation coordinates (`rank × tokens`) and parameter momentum (`rank × width`) without a unique mapping. We choose **row-any**: a down-projection row is eligible if any collected training token reaches `abs(A)>=pi/(2*kappa)-delta`. Across accumulation microbatches the eligible rows are unioned. Within these rows, only down-weight elements whose current gradient sign differs from their old Adam first-moment sign are reset. Up-projection momentum is not reset.

Reset happens **before standard AdamW accumulation**, following appendix A.13. Thus after the optimizer step the affected first moment is `(1-beta1)*gradient`, not zero. Equation 7's literal post-accumulation-zero reading would define a different optimizer. We use equation 6's unbounded outer interval, not the bounded interval mentioned elsewhere in A.13. These are declared implementation choices, not resolved author intent.

```python
adapter = OPSAAdapter(width=768)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.2e-4, weight_decay=.01)
optimizer.zero_grad()
adapted_tokens = adapter(tokens, collect_mapc=True)
loss = criterion(head(adapted_tokens[:, 0]), labels)
loss.backward()
torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
reset_count = adapter.reset_mapc(optimizer)
optimizer.step()
```

The snippet illustrates adapter usage only; include trainable head parameters in the actual optimizer and clipping set. With multiple adapters, call each adapter's reset before one optimizer step. Only collect valid training tokens, never query/evaluation forwards. Start each new training step with cleared observations; reset consumes them. If a step is skipped or aborted, call `clear_mapc()` and clear gradients. Use standard minimizing AdamW. For the no-MAPC control, omit collection and reset entirely.

CPU float32/float64 are tested. AMP, distributed aggregation, torch.compile, and checkpoint-recomputed forward tracking have not been validated; do not silently deploy this implementation with them. A nonfinite gradient raises before any reset, consumes pending rows, and requires skipping the optimizer step.

## Verification and experiment boundary

`python test_opsa.py` uses synthetic CPU tensors only. Tests cover explicit-matrix forward parity, quadrature orthogonality/isometry/J²=-I, Frobenius radius, parameter count, finite parameter gradients, input finite-difference gradcheck, serialization parity, selective reset, unchanged second moments, post-reset Adam accumulation, microbatch union and invalid-input handling.

No ISIC/HyperKvasir fit, encoder pass, real query scoring, or clinical-performance claim is included in this implementation delivery. It does not resurrect stopped configurations or recreate CLIP assets. Before a performance run, separately seal insertion sites, data isolation, matched initialization/training schedules, original F, linear adapter and no-MAPC controls, minority/all-class/fold-cohort gates and independent resource limits. HK E and Qgate are two query cohorts, not two producer folds.
