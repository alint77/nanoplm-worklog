# Pre-registration

The rules, fixed before any arm runs. The plan itself is in
`ablation-ladder.md`.

The point of writing these down first is that they cannot be adjusted after
seeing the numbers. That is the specific failure of the jul30 series.

## Metric

**Deciding:** biotrainer PBC and PGym, run later on the stored checkpoints.

**Secondary:** post-decay eval loss. Diagnostic only. It is scored at a fixed
masking recipe (15%, 80/10/10, token) with a fixed mask seed, decoupled from
whatever the arm trains at, so arms that change the training masking recipe
are still scored on the same task.

**Not deciding:** MFU, step time, throughput. All diagnostic.

## Budget

6 h stable phase, 4 nodes, 16 GH200. About 96 GPU-hours per run.

The 30 min decay is a separate campaign run later off the saved checkpoints,
not chained to the stable job. Add ~8 GPU-hours per arm when it happens.

## Noise floor

3 seeds plus 3 same-seed repeats of the base, at the LR chosen by the Tier 1
sweep. Gives sigma_seed and sigma_repeat.

## LR policy

NorMuon runs have two learning rates. `muon_learning_rate` covers the matrix
parameters; `adam_learning_rate` covers embeddings, the tied head, norms and
biases.

We sweep `muon_learning_rate` only, and hold `adam_learning_rate` at the value
the AdamW sweep picked. Sweeping both would be a 2D search for a group that is
small at vocab 32.

## Decision rule

An arm's best-over-LR result versus the base's best-over-LR result. The arm
wins only if it is better by more than **2 x sigma_seed**. Inside that band it
is reported as "no measured effect", not as a small win.

## Seeds

Tier 2: 1 seed per arm per LR, except our own ideas (2c), which get 2.
Tier 3 and 4: 2 seeds.

## What is frozen across all arms

Corpus (UniRef50, packed varlen, 512 tokens), 1,048,576 tokens per step, bf16,
FSDP2 per-layer with fp32 reduce, `num_workers: 4`, `eval_steps: 500`, profiling on, and
the pinned code SHA.

## Changed after the fact

Nothing yet. Any change to this file after Tier 0 starts gets a dated entry
here saying what changed and why.

## Amendment, 2026-09-10: fixed-step checkpoints for downstream

Wall-clock matching leaves arms at different step counts, and downstream scores
are read off whatever checkpoint an arm ended on, so node speed leaks straight
into them (r = 0.89 among identical configurations; see `findings.md`).

Every arm from Tier 2 on writes an extra checkpoint at **step 10500**, the
common step already used for the loss column. Downstream comparisons within a
tier use that checkpoint, not the wall-clock one. Arms already run (t0, t0b,
t1a, t1b, t1c) have no such checkpoint; their downstream numbers are reported
with the step beside them and are not used to separate arms whose loss gap is
under 0.010.

