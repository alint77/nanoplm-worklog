# Findings

Things that cost a run, or would have.

## FSDP silently turns fp32 layer inputs into bf16

`MixedPrecisionPolicy.cast_forward_inputs` defaults to `True`. Every
floating-point tensor passed to a wrapped module's forward is cast to
`param_dtype`. With per-layer sharding that is every encoder layer.

So any "this is fp32" assumption at a layer boundary is wrong today. Values in
[-1, 1] like cos/sin do not care. Raw angles do: at position 8191 the angle is
about 8191 radians, and bf16's 0.4% relative error is tens of radians.

Fix: hold such tensors as buffers. Buffers are not forward inputs.

Still to check: the residual-lambda tensors are passed as forward args, so they
are bf16 right now. Resid lambdas is an arm, so confirm before it runs.

## TE fused RoPE is faster and buys nothing

4 nodes, h960/L30:

| | eager | TE fused |
|---|---|---|
| rope kernel time (8 steps) | 486.4 ms | 260.9 ms |
| NCCL time | 627.8 ms | 1704.3 ms |
| step time | 470.5 ms | 470.6 ms |

Rope was already overlapping communication, so it was never on the critical
path. Removing work that is not the bottleneck moves nothing. It was neutral at
h960, the width where the eager path is worst, so it cannot help at h1024.

## TE's RoPE cost scales with the table length, not the sequence length

Same call, 128x512 packed, 15 heads:

| table rows | time |
|---|---|
| 512 | 0.287 ms |
| 2048 | 0.388 ms |
| 8192 | 1.274 ms |

Sizing the table at `max_position_embeddings` while running 512-token sequences
cost 4.4x and made the step 49% slower.

## A silent fallback is invisible in a profile

The compiled eager rope is named
`triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_view_*`. So "no TE
kernels in the trace" and "TE kernels present" look the same by absence.

Our first A/B reported a clean null result while the fused path had never run
once. Anything with a fallback has to say out loud which path it took.

## Two host syncs per micro-step

`num_valid_tokens` was a Python int. `_move_batch_to_device` only moves
tensors, so it survived to the forward and became a CUDA tensor there: a
blocking copy from pageable memory, one `cudaStreamSynchronize` per micro-step.
The non-finite check did an `all_reduce().item()` per micro-step as well.

Fixed both. 1 node -4.6%, 4 nodes -1.3%, host-blocked time to zero.

## Width sets MFU, not depth

Changing depth 21% (L28 to L34 at h896) moved MFU 0.2 points. Changing width
moved it 5. Comms share was flat and exposed comms was actually lower at the
larger size.

Also: only non-overlapped comms is a cost. Adding up total comms time is wrong.

## sbatch had no --nodes default

A job silently ran on 1 node instead of 4. The pipeline compensated by raising
grad_accum to 4, so it completed normally and reported 53.88% MFU, which was
not comparable to anything. Nothing in the log said "1 node".

Fixed with `#SBATCH --nodes=4`. Same class of bug as `num_workers: auto`.

## The nvidia-smi sampler starved the training step

A plain `srun` step holds the node's resources, so the training step sat in
"step creation temporarily disabled" for 19 to 32 minutes. Needs
`srun --overlap`.

## The VRAM log line was misread

`peak=67,950/69,024MB` is peak allocated over peak reserved, both from the
caching allocator. It is not used-over-total. The card is 97,280 MiB. The log
now prints the card total too.

## Eval is not comparable across MLM-rate arms, and it is noisy

Two separate problems, both open:

1. The eval collator uses the training `mlm_probability`
   (`pure_pipeline.py:2257`). An arm that trains at 15% is also evaluated at
   15%, so its loss is not comparable to a 30% arm. Different task.
2. Eval masks are redrawn every call from global RNG. No generator anywhere in
   `collator.py`. So eval loss carries fresh mask noise, and depends on how
   much RNG training consumed.

Both fixed in `4e3bbfc`. Eval now masks at a pinned 15% token / 80/10/10
recipe with a fixed seed, independent of what the arm trains at, and an eval
batch's mask is a hash of its own token ids so it is stable across runs,
worker counts and batch order.

The keys are `eval_mlm_probability`, `eval_mask_replace_prob`,
`eval_random_token_prob`, `eval_keep_probability`,
`eval_mlm_masking_strategy` and `eval_mask_seed`. All default to None, which
means "same as training", so nobody else's runs change.

## Eval costs 4.0 s a time

Measured off a real 4-node log, not estimated. At `eval_steps: 250` over a 6 h
run that is about 169 evals, roughly 11 minutes, 3% of the budget. Now at 500.

## A run has no final eval unless the step count lines up

Eval fires on `global_step % eval_steps == 0 or at_wsd_stable_end`, and there
is no eval after the training loop. So a decay run whose `decay_steps` is not a
multiple of `eval_steps` finishes with no score at all. Round it.

## NorMuon's LR scaling is shape-blind under rms_norm

`muon_adjust_lr` picks how the orthogonalized update is rescaled per matrix:

- `rms_norm` = `lr * 0.2 * sqrt(max(fan_out, fan_in))`
- `spectral_norm` = `lr * sqrt(fan_out / fan_in)`

`rms_norm` takes the max of the two fans, so it cannot distinguish a tall
matrix from a wide one. Consequences at our shape:

- the MLP down-projection `(1024, 2688)` gets 1.6x the square-matrix LR under
  rms_norm and 0.62x under spectral. A 2.6x difference in relative weighting.
- under GQA, the k-block `(512, 1024)` gets exactly the same LR as a full
  `(1024, 1024)` block under rms_norm. Under spectral it gets 0.707x.
- scaling h1024 to h1152 drifts every LR +6.1% under rms_norm and -0.8% to
  0.0% under spectral.

dion's docstring: spectral_norm is "for learning rate transfer across model
scale", rms_norm is "for learning rate compatibility with Adam/AdamW". dion
defaults to spectral_norm everywhere; nanoplm overrode it to rms_norm to
preserve pre-jul30 behaviour.

Also worth knowing: an LR tuned under rms_norm sits about 6.4x lower than the
equivalent under spectral_norm for square matrices. Reusing the number across
the two scalings sweeps the wrong decade.

## A wall-clock stop does not write "checkpoint-stable-end"

It writes `checkpoint-<step>` and says so:

    Stopped on the wall-clock budget at step 920/100000; skipping the
    terminal 'stable-end' checkpoint. Latest state is checkpoint-920.

The terminal role name is reserved for a run that reaches the end of its LR
schedule. A wall-clock stop never does, and naming a mid-schedule snapshot
"final" would misrepresent it and could clobber a real one in the same output
directory.

Harmless once you know: the checkpoint is complete and
`ResumeConfig.checkpoint_dir` takes an explicit path.

## dt in the log is a windowed average and includes eval

Step-time distribution over a 920-step run at `eval_steps: 50`:
`min 506, p25 512, median 514, p75 706, max 1796`.

The median is the real step time. The 706 values are logging windows that
contain an eval, and 1796 is the first window, which contains compile. Reading
the last line of a log as "the step time" will be wrong whenever that window
happened to include an eval.

## Held jobs do not start themselves after maintenance

The Tier 1 jobs were submitted just before a cluster-wide maintenance window
and ended up `PENDING` with `Reason=JobHeldUser` and `Priority=0`, which is a
hold, not a queue position. A held job stays held after the reservation ends.

So a submission that straddles a maintenance window needs an explicit
`scontrol release <jobids>` afterwards. `squeue` showing PENDING is not enough
to conclude a job will eventually run: check the Reason field.
