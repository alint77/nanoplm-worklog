# Pre-registration

The rules, fixed before any arm runs. The plan itself is in
`02-ablation-ladder.md`.

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
into them (r = 0.89 among identical configurations; see `04-findings.md`).

Every arm from Tier 2 on writes an extra checkpoint at **step 10500**, the
common step already used for the loss column. Downstream comparisons within a
tier use that checkpoint, not the wall-clock one. Arms already run (t0, t0b,
t1a, t1b, t1c) have no such checkpoint; their downstream numbers are reported
with the step beside them and are not used to separate arms whose loss gap is
under 0.010.

## Amendment, 2026-09-10 (2): fixed-step for compute-neutral arms

Supersedes the fixed-step-checkpoint amendment above, which patched the symptom
rather than the cause.

**The rule.**

- An arm that **changes compute per step** (architecture: MoE, canon layers,
  mHC-lite, QK-norm, GQA, attention pattern, activation, width, depth) is
  **wall-clock matched**. That is the whole point: a slower architecture gets
  fewer steps in six hours, and paying that cost is what makes the comparison
  honest. A faster one gets more.
- An arm that **does not change compute per step** (learning rate, weight
  decay, betas, cautious/nesterov, masking rate, masking split, seed, decay
  shape) is **fixed-step**: `max_steps: N`, with `max_wallclock_hours` left in
  place only as a safety net.
- Inside an architecture tier, the two combine: **the LR sweep for each
  architecture variant is fixed-step, and the comparison run between variants
  is wall-clock matched.** Tuning is a compute-neutral question asked within
  one architecture; the comparison is not.

**Why.** Wall-clock matching buys fairness only when the arms differ in cost.
When they do not, it buys nothing and injects node-speed variance into every
comparison. Concretely, on arms with no compute overhead it left checkpoints
spanning 672 steps, and that step spread correlated with downstream long P@L at
**r = 0.89 across six identical configurations**, accounting for 54% of what
was being reported as the downstream noise floor. Fixed-step removes it at the
source: every arm ends at the same step, so the loss needs no equal-step
correction and the final checkpoint is a common-step checkpoint by
construction.

**Cost.** Usually lower, not higher. The Tier 1a NorMuon curves show the LR
ranking is established by step 1500 and the best-to-second gap *peaks* around
steps 3500-5000 (0.0034-0.0042, 3-4x threshold) then **shrinks** to 0.0007 by
step 10500 as the top two converge. A short fixed-step sweep therefore
separates learning rates better than a full-budget one. Tier 1d runs at
`max_steps: 5000`: 40 node-hours instead of 96.

**Two things to get right when applying it.**

1. Size the Slurm limit from the *slowest* plausible step time, not the mean.
   At 1.99 ms/step observed worst case, 5000 steps is 2.8 h; request 4 h. The
   launcher hardcodes `--time=07:00:00`, so pass
   `SBATCH_EXTRA="--time=..."` to `tools/submit_gated.sh` (added
   2026-09-10) rather than over-requesting, which blocks backfill.
2. An LR picked at a short fixed step is picked under a **constant** LR, with
   no decay. Stable-phase selection is biased toward lower LRs, because a
   higher LR cashes in more during cooldown. Acceptable for ablations; the
   final long run must re-check the top two LRs with the decay attached.

