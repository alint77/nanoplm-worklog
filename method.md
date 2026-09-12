# Method

The rules, fixed before the arms run. The point of writing them down first is
that they cannot be adjusted after seeing the numbers. That is the specific
failure of the jul30 series.

## What decides

**Deciding:** biotrainer autoeval on the stored checkpoints. The reported set,
fixed by the team on 2026-09-11, is PGYM Spearman (total), zero-shot contact
prediction, supervised contact prediction, NewPISCES365, and subcellular
localization, in that order of priority. Of what we can run today,
**long-range contact P@L on `selected_protein`** is the deciding number (see
"How sharp each metric is"). Supervised contact, NewPISCES365 and subcell are
not wired yet: see [plan.md](plan.md#eval-work-the-new-metrics-need).

**Secondary:** eval loss. Diagnostic, but the only signal available while runs
are in flight. Scored at a fixed masking recipe (15%, 80/10/10, token) with a
fixed mask seed, decoupled from whatever the arm trains at, so arms that change
the training masking are still scored on the same task.

**Not deciding:** MFU, step time, throughput.

## Matching: fixed step or wall clock

- An arm that **changes compute per step** (MoE, canon, mHC-lite, QK-norm, GQA,
  attention pattern, activation, width, depth) is **wall-clock matched**. A
  slower architecture gets fewer steps in six hours and pays for the slowdown
  itself. That is what makes the comparison honest.
- An arm that **does not** (learning rate, weight decay, betas,
  cautious/nesterov, masking rate, masking split, seed, decay shape, rope theta)
  is **fixed-step**: `max_steps: N`, with `max_wallclock_hours` left as a safety
  net only.

**What that safety net is for, and how to set it.** On a fixed-step arm the
wall-clock stop does no matching work, so it is tempting to push it out of the
way. Do not: `slurm/sbatch_run.sh` has no `--signal` and the pipeline installs
no SIGTERM handler, so a job that reaches Slurm's `--time` is killed with **no
terminal checkpoint** and the whole run is lost rather than shortened. The stop
is what turns that into a short but usable arm, compared at the lowest step any
arm reached, which is already how the noise-floor replicates are read.

Set it *below* the Slurm limit, allowing for compile and the terminal save:
**6.5 against a 7 h allocation**. It counts steady-state training only (the
clock starts two steps after the run begins, excluding compile), so a value
equal to the Slurm limit can never fire. The Tier 2a rope arms were submitted at
7.0 against `--time=07:00:00` for exactly that reason, which silently disabled
the net; at the observed 1.97-2.05 s/step they project 5.9-6.1 h and were left
alone, but the margin was luck rather than design. Break-even for a 10500-step
arm on a 7 h allocation is 2.35 s/step, about 19% slower than nominal.

The node-health gate is worth keeping on fixed-step arms, but for a smaller
reason than on wall-clock ones: a slow node group no longer corrupts the
comparison (same steps, same data, same math, just later), it only risks the
Slurm limit above and wastes cluster time. The r=0.89 node-speed leakage that
justifies the gate is a wall-clock-matching artifact and does not apply here.
- Inside an architecture tier the two combine: the per-variant LR sweep is
  fixed-step, the comparison between variants is wall-clock. Tuning is a
  compute-neutral question asked within one architecture; the comparison is not.

**Why.** Wall-clock matching buys fairness only when arms differ in cost. When
they do not, it injects node-speed variance into every comparison. On
compute-neutral arms it left checkpoints spanning 672 steps, and that spread
correlated with downstream long P@L at **r = 0.89 across six identical
configurations**, accounting for 54% of what was being reported as the
downstream noise floor. Measured directly: three byte-identical runs have
sigma 0.00010 at equal steps and **0.00130 at their wall-clock stops**, 13x
noisier for the same runs.

**Cost: usually lower.** The Tier 1a NorMuon curves establish the LR ranking by
step 1500, and the best-to-second gap *peaks* around steps 3000-4000 then
shrinks as the top two converge. A short fixed-step sweep separates learning
rates better than a full-budget one, not just more cheaply.

**LR sweeps run at `max_steps: 5000`**, about 2.6 h per run on 4 nodes, so
request a 4 h Slurm limit. Anything shorter is a judgement call rather than a
saving: measured on the Tier 1d grid, the best-to-second gap goes 0.0017 at step
2000, 0.0034 at 2500, 0.0037 at 3000, 0.0036 at 3500, 0.0035 at 4000, 0.0026 at
4500, 0.0028 at 5000. The plateau is 3000-4000, so a 3500-step sweep resolves
the ranking about as well for 30% less wall clock, and that is worth knowing if
the queue is ever the bottleneck. 5000 is the standard because it costs little
here and leaves more margin on a shallow curve.

Two things to get right:

1. Size the Slurm limit from the *slowest* plausible step time. At 1.99 ms/step
   worst case, 5000 steps is 2.8 h, so request 4 h. The launcher hardcodes
   `--time=07:00:00`, so pass `SBATCH_EXTRA="--time=..."` to
   `tools/submit_gated.sh` rather than over-requesting, which blocks backfill.
2. An LR picked at a short fixed step is picked under a **constant** LR.
   Stable-phase selection is biased low, because a higher LR cashes in more
   during cooldown. Fine for ablations; the final long run must re-check the top
   two LRs with decay attached.

Arms already run under wall-clock matching (t0, t0b, t1a, t1b, t1c) are read at
a common step of **10500** for loss. Downstream scores have no such correction,
so read the step beside them and never use them to separate arms closer than
0.010 in loss.

## Noise floor

Six runs at the decided base, NorMuon 7e-3: three same-seed repeats and three
extra seeds. All six passed the trace gate first (exposed/compute 0.5% to 5.5%,
threshold 8%), with the 8 known-bad nodes excluded.

| run | seed | steps | wall-clock | equal-step (10500) |
|---|---|---|---|---|
| t0-rep1 | 42 | 11000 | 2.2041 | 2.2065 |
| t0-rep2 | 42 | 10500 | 2.2064 | 2.2064 |
| t0-rep3 | 42 | 10500 | 2.2063 | 2.2063 |
| t0-seed43 | 43 | 10500 | 2.2077 | 2.2077 |
| t0-seed44 | 44 | 10500 | 2.2058 | 2.2058 |
| t0-seed45 | 45 | 10500 | 2.2078 | 2.2078 |

| quantity | equal-step | wall-clock |
|---|---|---|
| sigma_repeat (3 identical runs) | **0.00010** | 0.00130 |
| sigma_seed (4 seeds) | **0.00097** | 0.00176 |
| **2 x sigma_seed** | **0.0019** | 0.0035 |

sigma_seed is ~10x sigma_repeat, so run-to-run variation is dominated by data
order rather than kernels, which is the healthy ordering. An earlier estimate
from three accidental duplicate pairs in Tier 0b put sigma_repeat at 0.0004
(2.3086/2.3089, 2.3100/2.3106, 2.3248/2.3250); the Tier 0 number supersedes it.

**Sample-size caveat.** sigma_seed comes from n=4, so the 95% interval on sigma
runs roughly 0.55x to 2.9x the estimate: 2 sigma could plausibly be anywhere
from 0.0011 to 0.0056. That does not threaten the large wins, and it is why a
-0.0050 result is adopted but flagged.

## Decision rule

An arm's best-over-LR result against the base's best-over-LR result. The arm
wins only if it is better by more than **2 x sigma_seed = 0.0019** at equal
steps. Inside that band it is reported as "no measured effect", not as a small
win. Best-over-LR is optimistic on both sides, which is why the base gets the
same treatment.

## How sharp each metric is

The contact framework reports 48 metrics: 3 datasets x 4 separation bands
(`local` [3,6), `short` [6,12), `medium` [12,24), `long` [24,-)) x 4 readouts
(AUC, P@L, P@L2, P@L5). PGYM adds scc and ndcg. Within a band the four readouts
are one ranking read four ways, so the informative axes are dataset and band.

**Aggregation is development mode from 2026-09-11.** Development mode subsamples
the benchmark (casp14 38, casp15 38, selected_protein 71, PGYM 86 assays of 217)
with a fixed seed, so it is the same proteins every time. It costs 1.7x on the
deciding metric's resolution and buys about 5x in GPU time (31.5 min against
2.7 h per checkpoint):

| aggregation | n | 2 sigma (step-corrected) | resolvable loss gap |
|---|---|---|---|
| full | 1430 | 0.0022 | 0.0011 |
| development | 71 | 0.0038 | 0.0019 |

That is far less than sampling theory suggests (sqrt(20) would be 4.5x), because
replicate variance is dominated by model-level differences correlated across
proteins rather than by per-protein noise. The floor stays level with val loss.

Arms already scored in full mode do not need rescoring: the per-protein
Jacobians are cached, so `eval/tools/dev_aggregate.py` re-aggregates any of them
over the same development subsets for free. Dev-mode and full-mode numbers are
**not** interchangeable (the base mean moves 0.3930 to 0.3955), so a table must
pick one and say which.

**The rule for picking the deciding metric.** For each metric, 2 sigma over the
six base replicates divided by its slope against eval loss gives the loss gap it
needs before it can call a winner. The deciding metric is the smallest such gap
on `selected_protein`, the only dataset with enough proteins (n=1430 against
n=96 and n=95 for the CASP sets) to resolve arms, with ties inside 10% broken
toward the conventional P@L. Run on all 50 by `eval/tools/resolution.py`:

Development-mode aggregation, step-corrected, from
`eval/tools/dev_aggregate.py`:

| metric | 2 sigma | slope | loss gap needed |
|---|---|---|---|
| **selected_protein long P@L** | 0.0038 | -1.997 | **0.0019** |
| selected_protein medium P@L2 | 0.0038 | -1.249 | 0.0030 |
| selected_protein long AUC | 0.0073 | -2.288 | 0.0032 |
| selected_protein long P@L5 | 0.0087 | -2.475 | 0.0035 |
| selected_protein long P@L2 | 0.0100 | -2.366 | 0.0042 |
| selected_protein medium P@L | 0.0048 | -0.792 | 0.0061 |
| casp14 long P@L | 0.0108 | -0.853 | 0.0127 |
| selected_protein local P@L | 0.0066 | -0.511 | 0.0129 |

**long P@L wins outright in development mode**, where in full mode it was a
three-way tie with long AUC and long P@L2 that needed the tie-break. The AUC and
P@L2 readouts degrade faster at n=71, which is a small piece of luck: the metric
the rule already chose is the one that survives subsampling best.

The pattern is the same in both aggregations. The sharpest metrics are all
`selected_protein long`, every `local` metric is the bluntest of its dataset, so
the ordering is a property of the separation band and not of the readout. Long
contacts are as sharp as val loss (0.0019) and several times sharper than PGYM.
Not because they are quieter, the noise is comparable, but because they respond
several times more steeply.

Full-mode figures for the same rule, kept for the 67 arms scored that way:
long P@L 2 sigma 0.0048 raw and 0.0022 step-corrected, slope -1.958, gap 0.0011;
long AUC 0.0023; long P@L2 0.0024; medium P@L 0.0041; pgym scc 0.0040 / -0.438 /
0.0091; casp14 local P@L 0.0179; worst of 48 was casp14 local P@L5 at 0.0328.
The raw-versus-corrected split matters because the step-count leak inflates the
raw floor by 2x (see
[findings.md](findings.md#wall-clock-matching-contaminates-downstream-scores-through-step-count)).

PGYM's own replicate sigma is 0.0018 over three same-seed repeats (2 sigma
0.0037), 0.0024 over three seeds (2 sigma 0.0047), 0.0020 pooled. It confirms
large decisions and cannot adjudicate small ones: a 0.001 loss gap (20% vs 25%
masking) is invisible to it.

**One honest caveat about the selection.** A metric chosen for how steeply it
tracks eval loss will, by construction, tend to agree with loss-based verdicts,
so "two independent measures, same answers" reads stronger than it is. It does
not threaten the large calls, which clear any reasonable metric by 4x to 17x.
Where downstream actually earns its keep is where it *disagrees* with loss: the
100/0/0 masking arms, and the `t0b-normuon-b1M-1721969` collapse. Keep PGYM
reported alongside for that reason, since it is constructed differently
(masked marginals on one position, no random tokens) and can dissent.

## Budget and what is frozen

6 h stable phase, 4 nodes, 16 GH200, about 96 GPU-hours per run. The 30 min
decay is a separate campaign off the saved checkpoints, not chained to the
stable job; add ~8 GPU-hours per arm when it happens.

Frozen across all arms: corpus (UniRef50, packed varlen, 512 tokens), 4.19M
tokens per step, bf16, FSDP2 per-layer with fp32 reduce, `num_workers: 4`,
`eval_steps: 500`, profiling on, and the pinned code SHA.

**LR policy.** NorMuon runs have two learning rates: `muon_learning_rate` for
the matrix parameters, `adam_learning_rate` for embeddings, the tied head, norms
and biases. We sweep the muon one only and hold the adam one at what the AdamW
sweep picked. Sweeping both would be a 2D search for a group that is tiny at
vocab 32.

**Seeds.** Tier 2: 1 seed per arm per LR, except our own ideas (2c), which get
2. Tier 3 and 4: 2 seeds.

## Overrides, logged

**2026-09-11: masking rate set to 15% on downstream evidence, against the loss.**
Eval loss put the optimum at 20-25% and ranked 15% +0.0065 worse. Both
downstream metrics put it at 15-20%: 15% is the best 80/10/10 cell on long P@L
(0.4149 corrected) and the best rate on PGYM. The pre-registration names
downstream as deciding and loss as diagnostic, so this follows the rule rather
than overriding it, but it is the first decision where the two measures point
different ways and the loss lost, which is worth a dated line.

**2026-09-11: rmsnorm, swiglu and QK norm adopted without an arm each.** A
baseline modernization on citation and kernel-support grounds, not a finding of
this series (reasoning in
[decisions.md](decisions.md#the-modernized-baseline-rmsnorm-swiglu-qk-norm)).
The bundle is measured against stock ModernBERT in one wall-clock-matched run,
so the paper has a number for the modernization as a whole; it is not
decomposed, and the writeup must not imply otherwise.

**2026-09-11: Tier 4 and Tier 5 struck.** Leave-one-out and the 600M transfer
plus fp8 pair are out of scope. Consequences to state in the paper: the final
recipe is not leave-one-out validated, so it may contain ingredients that only
measure each other, and nothing is checked at a second scale, which the batch
decision explicitly flagged as owed ("chosen at 400M, the 600M transfer has a
different critical batch").

**2026-09-11: the modernization LR grid was not extended, against the standing
rule.** The rule says a winner at the edge of the grid means extend before
building on it, and that is what happened to the first 5-point grid. The 7-point
relaunch also put its best point at the edge, 2.8e-2, so the rule asks for 4e-2.
It was not run. The reason the rule exists is a winner that might really sit
outside the grid; here the top two points differ by 0.0008 against a resolution
of 0.0019, and the last three gaps are 0.0022, 0.0017, 0.0008, so the curve is
flat at the top rather than still climbing. 2e-2 was taken, the lower of two
indistinguishable points. Cost if this is wrong: the modernization long run,
which is the headline comparison against stock ModernBERT, gets an LR up to one
grid step below optimal, which would understate the modernization. Reopening it
costs one 2h45 sweep arm at 4e-2 plus a rerun of the long run.

**2026-09-12: supervised contact is the deciding contact readout.** Team call,
resolving the ambiguity logged below. It is the better-supported choice on the
evidence: on the same checkpoints the supervised readout produced a smooth
monotone curve across the rope ladder (Spearman 0.900 against log M, zig-zag
0.0007) while the zero-shot Jacobian zig-zagged 0.0112, sixteen times as much,
and the Jacobian is also the readout that came apart from the probe by 0.0485 on
the modernized base.

Two consequences to carry into the writeup. **The modernization now reads as a
win**: supervised contact is +0.0371 on stock ModernBERT where zero-shot was
-0.0161, so the adopted rms+swiglu+qknorm base is better on the deciding metric,
better on loss and better on PGYM, and worse only on the readout no longer
deciding. **And every decision taken before today rests on the old decider** --
the masking rate, the masking split and the Tier 1 optimizer and batch
conclusions were all read on zero-shot long P@L. They are not being re-derived;
anyone re-reading them should know which metric was in force.

**2026-09-12: QK norm is not being tested.** Team call. The Jacobian regression
on the modernized base therefore stands unexplained: parameter-free
`F.rms_norm(q, (head_dim,))` caps attention logits at `+-sqrt(head_dim)`, which
remains the leading hypothesis with no arm behind it. Cost of leaving it: the
paper cannot say why the two contact readouts diverge on this architecture, and
the candidate fix (a learnable per-head scale, as in OLMo-2 and ViT-22B) is
untried. Cheap to revisit later, one arm of `base-modern15` with
`use_qk_norm: false`.

**2026-09-12: the deciding metric was ambiguous once both contact readouts
ran.** The pre-registration named zero-shot long P@L on `selected_protein` as
deciding, and it was chosen when supervised contact could not run at all. On the
modernized base the two readouts disagree in opposite directions, past 5 sigma
each ([findings.md](findings.md#the-modernized-base-decouples-the-two-contact-readouts)).
The rule as written does not say which wins, so it cannot settle whether the
adopted base is better or worse than stock ModernBERT. Flagged to the team
rather than resolved here, because picking the readout after seeing the result
is exactly what pre-registration exists to prevent. Note for whoever settles it:
the probe reads frozen features, the Jacobian reads the model's own output
sensitivity, and only the latter regressed.

**2026-09-11: the masking split cleared the rule and was not adopted.**

What cleared: (20%, 100/0/0) beats (20%, 80/10/10) by 0.0127 on the deciding
metric, 5.8x the 0.0022 threshold, and by a consistent margin on 16 of 20
per-band comparisons. The pre-registered rule says adopt.

Why it was deferred: the 10% random tokens in 80/10/10 train the model to be
robust to substitutions, and the categorical Jacobian measures substitution
sensitivity, so the benchmark may reward the arm that is twitchier rather than
the one with better structure. Every zero-shot metric in this series reads the
model's own output sensitivity, so none of them can separate the two readings.
That is a named mechanism, not a general doubt, and PGYM cannot arbitrate: the
effect is 0.0028 scc against its 0.0040 threshold.

What resolves it: a probe trained on frozen features, plus two confirmation
seeds per arm at step 10500. Both are in
[plan.md](plan.md#next-settle-the-masking-split). If the probe agrees, the base
moves to 100/0/0 and this entry records a delay rather than an override. If the
gap vanishes, the zero-shot contact metric cannot referee masking-split arms at
all, and that belongs in this file.

Deferring a result that cleared the rule is a deviation either way, which is why
it is written down here with a date rather than settled quietly in a results
file.

## How the matching rule evolved

Superseded, kept for the record. The first amendment (2026-09-10) had every arm
from Tier 2 on write an extra checkpoint at step 10500 and compared downstream
scores there. That patched the symptom. The rule above, from later the same day,
removes the step spread at the source instead. Any further change to this file
gets a dated entry here saying what changed and why.
