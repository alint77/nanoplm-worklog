# Plan

What is left to run. Done work is in [results.md](results.md); the matching and
threshold rules are in [method.md](method.md).

Every arm since 2026-10-05: 4 nodes, 16 GH200, 4.19M tokens/step (grad_accum
4), constant LR to step 13000 (54.5B tokens), then a 2000-step 1-sqrt cooldown,
read at both points. Compute-neutral arms are step-matched; arms with step-time
overhead get round(13000 / f) and round(2000 / f) steps, f measured once on one
allocation ([method.md](method.md#wall-clock-matching-by-a-measured-factor)).
Each arm changes exactly one thing from base v2 and gets a 3-LR fixed-step sweep
at 5000 steps first, extended at any edge; the top two LRs go to the long run.

## Roadmap (as of 2026-10-22)

Designed 2026-09-26, with the team's notes of 2026-09-29 and 2026-10-05
applied. The first sections below this one are the record of earlier stages.

| stage | what | status |
|---|---|---|
| 0 | measurement: noise floors per metric at 13k (n=7), eval battery, dev mode | done |
| 1 | shape (h1024 / L32, 399.6M) and batch (4.19M, confirmed at 44B) | done |
| 2 | optimizer recipe (NorMuon, wd 1e-5; knobs kept as measured at 30% masking) | done |
| 3 | modernization (rmsnorm, swiglu, QK norm; LR 2e-2 confirmed at 44B) | done |
| 3b | **base v2**: 3 seeds, 13k + decay | **done** 2026-10-07 ([results](results.md#base-v2-three-seeds-13k-and-the-decay-to-15k-2026-10-07)) |
| 4a | position encoding: rope theta (6 arms), all-global, at 13k + decay | **done, negative** ([results](results.md#position-encoding-at-13k-and-after-the-decay-2026-10-06)) |
| 4d | our own ideas (Tier 2c), 6 arms on **dense** base v2 | **done, nothing adopted** 2026-10-22 ([results](results.md#tier-2c-our-own-ideas-on-dense-base-v2-2026-10-06-to-10-14)) |
| 4b | MoE redo on base v2: pick the cell, which becomes base v3 | **void, redoing on tree v3** (2026-10-29): the expert layout bug (#170) gave experts the wrong LR scale and NorMuon axis, and the g7 NaN was our compiled bf16 Polar Express diverging (now GNS, [findings](findings.md#g7-s12-weight-nan-2026-10-27-optimizer-nccl-and-stray-write-audit-2026-10-28)). Old result kept for the record: g4-S12 2.0752 vs dense 2.1246. Redo: A-B-A factors (2576640), muon LR sweep at 5000 steps on dense {1.41,2,2.83}e-2 and g4-S12/g7-S12 {0.707,1,1.41,2,2.83}e-2 (2576641-53), then 4 cells + dense GNS control, trunk + decay |
| 4c | canon (kernel 7, mode x layer set), on the MoE base | **void, redo after 4b** on tree v3: canon LR sweep on arm C, then A, B, C, AC, ABC. Old: at canon LR 1e-4 all arms lost; sweep on C picked 1e-3 ([results](results.md#stage-4c-at-canon-lr-1e-4-no-arm-beats-base-v3-and-the-canon-lr-was-never-tuned-2026-10-28)) |
| 4e | Loopie and Huginn | held: loop designs undecided; Loopie needs PR #158 |
| 5 | greedy ladder + light leave-one-out, with the stage-2 knobs re-checked (AdamW-group LR 1e-2, wd 1e-5) | after 4b/4c; no Tier 2a/2c arm qualifies for it |
| 6 | second-scale check | struck by team decision |
| 7 | final model with decay, top-2 LRs, full-mode eval, against ESM-2 and ESM C | last |

**Order.** The 2026-09-29 call puts canon and our own ideas on top of the MoE
base. Our own ideas ran on dense instead (2026-10-06) so that work would not
wait on the PRs; none of them won there, so a re-check on the MoE base is only
owed for whatever the team still wants to carry.

**What tree v2 needs before 4b.** The frozen tree carries features that are in
no upstream commit (`fsdp_keep_unsharded_across_accum`, `moe_nan_guard`) and
three in-place patches (decay data position, the mHC port, the post-resume
`empty_cache`, see [env.md](env.md#frozen-tree-patches)). Tree v2 has to
contain all of them, and base v2 gets one parity run on it (same config and
seed, losses within run-to-run noise) before any MoE arm is compared against it.

**Housekeeping (done 2026-10-26).** The 2026-10-06 to 10-14 checkpoints on fscratch fall to the
one-month purge from about 2026-11-06. Archive what the paper needs (base v2
seeds, the Tier 2a and 2c endpoints and decays) to project1 before then.

## Done (2026-09-25): batch size at the real horizon, and optimizer follow-ups

**Batch size to 44B tokens.** Tier 0b compared batches at 10.07B tokens only
(a ~1.4 h token-matched probe, by design), and the 4.19M choice took its 0.0114
cost as an upper bound without ever checking it at the ~44B a real arm trains
for. Closing that now. The 4M point already exists at the real horizon: the
Tier 1a control `t1a-normuon-lr7e-3` has the identical training config to the
4M probe arm `t0b-normuon-b4M-lo2` (only eval cadence and stop rule differ, and
the two agree where they overlap: 2.3200 @2400 vs 2.3155 @2500), and reads
**2.2061 at step 10500 = 44.04B tokens** (2.1936 at 13000). So only 1M and 2M
need extending, each at its Tier 0b best LR:

| batch | arm | LR | segment 1 | segment 2 | segment 3 (to 44.04B) |
|---|---|---|---|---|---|
| 1.05M | `t0b-normuon-b1M-lo2` | 5e-3 | 9600 steps, 10.07B | to 19200, 20.13B | to **42000** |
| 2.10M | `t0b-normuon-b2M-lo3` | 5e-3 | 4800 steps, 10.07B | to 9600, 20.13B | to **21000** |
| 4.19M | `t1a-normuon-lr7e-3` | 7e-3 | (control) | | **10500** exists |

Both stop points are multiples of the arm's eval cadence (400 / 200 steps), so
all three are read at exactly 44.04B tokens, each with its own eval. The Tier 0b
terminal checkpoints predate the data-position fix; segment 2 resumes from
copies in `checkpoints-src-t0b20b/` whose `training_state.json` records the
reconstructed position (epoch 0, 9600 microbatches), and the resume logged
exactly that. Segment 2 runs (jobs 2004837-9, `checkpoints-t0b20b/`) also
re-run 4M to 20B, which duplicates the control's first 4800 steps and serves as
a repeat check only. **Segment 2 result (20.13B tokens): 1M 2.2608, 2M 2.2592, 4M 2.2615**, a 0.0023 spread against 0.0114 at 10.07B, so the large-batch penalty is shrinking as predicted. Segment 3 submitted 2026-09-25 as jobs 2005879 (1M) and 2005880 (2M), eval 2005881, `checkpoints-t0b44b/`. **Done: at 44.04B 1M 2.2169, 2M 2.2085, 4M 2.2061; 4M confirmed** (results.md). Downstream eval pending. Caveat to carry: the 1M/2M LRs were picked at 10B tokens,
and the optimum may drift lower over a longer horizon.

**Is the modernized base's 2e-2 still right at the full horizon?** It was
picked from a 5000-step (21B) sweep whose curve was still rising at the top,
and short constant-LR sweeps flatter higher LRs (Tier 1d: 1e-2 was second
through step 3000, 5e-3 from 3500 on). Every Tier 2/3 arm inherits it. Check:
resume the 5e-3 and 1e-2 sweep arms from step 5000 to 10500 (1.41e-2 dropped by user call; jobs
2011243 and 2011136, evals 2011244 and 2011138, `checkpoints-t2p105/`) and read against the 2e-2 arm's
continuation `t2-mod-long` at step 10500 (**2.1751**). All three sweep
checkpoints predate the data-position fix, so all three replay epoch 2 from the
top after step 5000: the data order is identical across the three, which makes
this an exactly matched comparison. If a lower LR wins at 44B, the base LR
moves before MoE and Tier 2c build on it. **Result: 2e-2 holds** (1e-2 +0.0048 at 9.5k, 5e-3 +0.0141 at 10.5k, both flat; decisions.md). Protocol change regardless: Tier 2c
extends each arm's top two short-sweep LRs to full length and picks there.

**A measured downstream noise floor at equal steps.** The Tier 0 floor is
equal-step for loss (2 sigma 0.0019 at step 10500) but not for downstream: the
six replicates were scored at their wall-clock stops (steps 10624-11036) and
statistically corrected to 10500 on six points, and supervised contact has
never had a floor at all. Resuming all six to step 13000 (jobs 2016168-74,
`checkpoints-13k/`) and scoring them on the full battery (PGYM, zero-shot and
supervised contact, secondary structure incl. newPISCES364, scl) gives a
directly measured floor for every column. The Tier 1a NorMuon control is the
identical config at seed 42, already scored at 13000, so it is a seventh
sample: 4 same-seed runs and 3 seeds.

**Optimizer follow-ups, fresh to step 13000** (jobs 2003624-7, eval 2003628),
all read against the 13k NorMuon control (2.1936):
- NorMuon's AdamW group (embedding, unembedding and all LayerNorm weights share
  one LR, 1e-4 today) at 3e-4, 1e-3, 3e-3.
- cautious WD + WD 1e-5 together: ~2.187 if they are one effect (wd1e-5 alone
  2.1870, cautious alone 2.1900), ~2.183 if additive.

## Done: the modernized base

Two things, in this order, before any Tier 2 arm runs.

**1. LR re-sweep on the new base, fixed-step at `max_steps: 3500`.** First wave
(jobs 1759347-1759351) returned only three arms: Slurm killed 7e-3 and 1e-2 on a
node failure and the launcher's retry was broken (see
[findings.md](findings.md#a-retrying-log-line-that-never-retried)). What came
back:

| LR | loss @3500 |
|---|---|
| 3.5e-3 | 2.2923 |
| 5e-3 | 2.2819 |
| **1.41e-2** | **2.2674** |

Monotone down to the top of the grid, so **QK norm moved the LR optimum up** and
the winner sits at an edge. Per the standing rule, the grid gets extended before
anything is built on it.

Relaunched 2026-09-11 as a single 7-point grid at `max_steps: 5000`, jobs
1761232-1761238. All seven COMPLETED, ~2h45 each. Eval loss at step 5000:

| LR | eval loss |
|---|---|
| 3.5e-3 | 2.2593 |
| 5e-3 | 2.2499 |
| 7e-3 | 2.2430 |
| 1e-2 | 2.2395 |
| 1.41e-2 | 2.2373 |
| **2e-2** | **2.2356** |
| 2.8e-2 | 2.2348 |

Still monotone to the top, but flattening hard: the last three steps of the grid
are worth 0.0022, 0.0017 and 0.0008. The final gap, 2.8e-2 over 2e-2, is 0.0008,
which is inside `2 * sigma_seed` = 0.0019 at equal step ([noise
floor](method.md#noise-floor)), so the top two points are not distinguishable.
Grad norms are flat across the whole grid (median 0.105 to 0.124, no spikes), so
nothing is unstable at the top; the grid is bounded by resolution, not stability.

**Decided: 2e-2**, taken over the nominally-best 2.8e-2 because the two are
within noise and the lower value is the safer of two indistinguishable points.
The grid was not extended to 4e-2; that is a logged override
([method.md](method.md#overrides-logged)).

Against the stock base this is a 3x move: 7e-3 was the interior optimum there
three times over, and here it is third worst at +0.0074.

Expected direction: QK norm bounds attention logits, which is what usually
limits the learning rate. The base changed twice at once: rmsnorm + swiglu +
QK norm, and masking 20% to 15%. The architecture change moves compute per step,
so the matching rule requires a fresh compute-neutral sweep before anything is
compared against it.

Sweep length is 3500 steps, the new standard: the Tier 1d gap data puts the
plateau at 3000-4000 and 5000 past it (see
[method.md](method.md#matching-fixed-step-or-wall-clock)). An earlier submission
of this same sweep at 5000 was cancelled 2 min in, during compile, and replaced.

Five points rather than Tier 1d's four, extending to 1.41e-2: QK norm stabilises
attention logits and can move the optimum up, and the Tier 1d curve already had
more room above 7e-3 (+0.0050 at 1e-2) than below (+0.0094 at 3.5e-3). The grid
brackets the optimum whichever way it moves.

Verified before submitting, on the config the runs actually use: rmsnorm reaches
the layers (`attn_norm` is `RMSNorm`), swiglu is parameter-neutral against geglu
(`mlp.Wi` stays 5376x1024), QK norm is consumed in the attention forward, and
the whole model is still **399.6M parameters**, so the modernization is
param-neutral and the comparison against stock is clean on that axis.

QK norm here is parameter-free `F.rms_norm(q, (head_dim,))` per head, applied
**after** RoPE (`reorder_RoPE_QKNorm: false`). Arm A5 would have tested the
order and is struck, so that ordering is now an untested choice in the base
rather than a measured one.

**2. The modernization long run (1 run, wall-clock matched).** The modernized
base at its winning LR against stock ModernBERT at the same 15% / 80-10-10.
**The control already exists**: `t1c-mlm15-m80-1739946` is stock architecture at
exactly that masking, 6 h wall-clock, loss 2.2014 at step 10500 and long P@L
0.4207 in dev-mode aggregation. So this is one run, not a pair, and it gives the
paper a single number for what the modernization is worth as a bundle.

Submitted 2026-09-11 as `t2-mod-long`, job 1762508, and it does not retrain the
first 5000 steps: it resumes the 2e-2 sweep arm's terminal checkpoint with
`resume.mode: continue`, which is what makes a 5000-step sweep affordable now
that it is half an ablation run. Wall-clock matching then works on the total
across both segments: the control achieved 21,617.9 s of steady-state training
(nominal 6 h), the sweep arm spent 9,879.0 s reaching step 5000, so the resumed
segment gets `max_wallclock_hours: 3.2558` (11,721 s). The clock excludes
compile and warmup at both ends, so the two runs are matched on steady-state
training time, which is what the rule asks for.

One asymmetry, logged rather than fixed: the sweep ran on the pinned tree, which
predates [PR #154](findings.md#terminal-checkpoints-did-not-record-the-data-position),
so its terminal checkpoint carries no data position and the resumed run restarts
epoch 2 from the top. It re-sees the 96 steps of epoch-2 data the sweep already
trained on, 1.6% of the run, on data the model has already been through twice.
Expected effect well under the 0.0019 resolution. The pinned tree now carries
the fix, so every later Tier 2c long run resumes cleanly.

## Closed: the masking split

**Closed 2026-09-25 at 15%, 80/10/10** (team decision). The seeds and probe below are no longer owed; kept as the record of what was considered.


Still open, and the 15% decision moved it rather than closing it. Downstream
reversed the loss on 100/0/0, which beats 80/10/10 on 16 of 20 band comparisons
([results](results.md#downstream-reverses-the-loss-on-pure-masking)). At the
chosen 15% rate the long band is a tie (-0.0006) but the near-range bands show
the **largest** 100/0/0 advantage anywhere in the grid: +0.0251 local, +0.0147
short, +0.0108 medium. So the base runs 80/10/10 while the evidence for the
alternative sits in the bands we are not deciding on.

Two things settle it, and the first is worth more than the second.

**1. Supervised contact**, once it is wired
([below](#eval-work-the-new-metrics-need)). Every zero-shot metric we have reads
the model's own output sensitivity, and the 10% random tokens in 80/10/10 train
exactly the substitution robustness the categorical Jacobian measures. A probe
on frozen features is the only thing that separates "better structure" from
"twitchier model". PGYM cannot referee it: the effect works out to 0.0028 scc
against its 0.0040 threshold, so it is underpowered by construction rather than
dissenting.

**2. Confirmation seeds, at 10500 steps and not 5000.** Two seeds each of
(15%, 100/0/0) and (15%, 80/10/10). The replicate band and threshold were both
measured at step 10500 and long P@L moves 1.26e-5 per step, so a 5000-step
checkpoint cannot be compared against either. About 380 GPU-hours.

Note throughout that eval loss cannot referee this comparison at all: the pinned
eval feeds 10% random tokens, so the 100/0/0 arms are scored on a task they
never trained for.

## Tier 2a: rope theta (6 runs), then all-global

Reordered 2026-09-11: rope theta goes first and gets six arms instead of one.
The original A7 was a single arm at `global_rope_theta: 10000`, which turned out
to be half a knob. The all-global arm (A6) keeps its config, generated and
unsubmitted, and runs after.

### Why six, and why both thetas

RoPE attention depends only on the relative offset `i - j`, so what matters is
how many of the 32 frequency pairs complete a cycle inside the largest offset a
layer can actually use: 512 for a global layer (max sequence length), **64 for a
sliding layer** (the window is +-64). Count those, and the shipped base is
badly over-provisioned at both scales, worse on the local layers, which are 21
of the 32:

| theta | longest wavelength | live dims, global (offset 512) | live dims, local (offset 64) |
|---|---|---|---|
| 160000 | 691,000 | 12/32 | |
| 10000 | 47,000 | 16/32 | 9/32 |
| 500 | 2,600 | 23/32 | |
| 60 | 335 | | 19/32 |

The remaining pairs rotate so little across the usable range that they are
effectively position-independent. So the hypothesis is two-sided, and that is
the reason to run a ladder rather than one point: **lowering theta trades
content-only capacity for positional capacity.** Those near-constant dims are
implicit NoPE dims, and NoPE dims are not obviously waste (`rope_dim_fraction`
exists to create them on purpose), so the curve could go either way and could
turn anywhere along it.

One rule sets both thetas: longest wavelength = `M` x the largest usable offset
for that layer type, i.e. `theta = (M * offset / 2pi) ** (64/62)`.

### What the field uses

Both checked in source, neither tunes theta to protein length and neither has a
local/global split:

- **ESMC**: `EsmcRotaryEmbedding(d_model // n_heads)`, no base override, so
  **10000** ([Biohub/esm](https://github.com/Biohub/esm),
  `esm/models/esmc/layers.py:437`).
- **AtlasLM**, the language model inside AtlasFold: `RotaryEmbedding(d_model //
  n_heads)`, also **10000**, `max_seqlen=20000`, and the file notes it started
  from ESM-3's implementation
  ([SeonghwanSeo/atlasfold](https://github.com/SeonghwanSeo/atlasfold),
  `src/atlaslm/layers/transformer_stack.py:103`).

So 10000 is inherited from text models, twice over, and everything below it is
untested. Arm A is that value on our FSS pattern; it is **not** a replication of
either model, both of which are all-global.

### The six arms

All hold the FSS pattern (11 global, 21 local) and LR 2e-2.

| arm | global | local | live g | live l | what it is |
|---|---|---|---|---|---|
| (control) | 160000 | 10000 | 12/32 | 9/32 | `t2p-mod-lr2e-2`, already run |
| A | 10000 | 10000 | 16/32 | 9/32 | the field's value; vs control = global-only |
| B | 10000 | 1200 | 16/32 | 11/32 | M=92; vs A = local-only |
| C | 2000 | 250 | 19/32 | 14/32 | M=18 |
| D | 500 | 60 | 23/32 | 19/32 | M=5 |
| E | 100 | 12 | 31/32 | 30/32 | M=1, the floor: every pair completes a cycle |
| F | 500 | 10000 | 23/32 | 9/32 | global-only at M=5; with D and control, a 2x2 |

The decomposition falls out of the ladder rather than costing arms: control->A
moves global alone, A->B moves local alone. F adds the same split at the
aggressive end. A local-only arm at M=5 (160000/60) completes the 2x2 and is
held back as one follow-up run if M=5 wins.

Not bracketed: every arm is at or below the base. If the base wins the ladder,
that is evidence for the content-only reading, and closing it needs one arm
*above* base (theta ~1e6). Held as a follow-up rather than spent now.

### Protocol

**No new control, and no LR sweep.** The arms inherit 2e-2 because RoPE is a
rotation: it preserves `||q||` and `||k||`, and QK norm is applied after it, so
the attention logit scale that sets the LR ceiling is untouched. This is a
stronger claim than compute-neutrality and is why these arms do not get the
3-LR treatment the all-global arm will.

Theta is compute-neutral, so per the matching rule these are **fixed-step, not
wall-clock**: six arms at `max_steps: 10500`, from scratch, one segment each.

10500 and not a cheaper 5000-step screen, which is what was first submitted and
then cancelled 8 minutes in. Every downstream calibration in this series sits at
that horizon: the six noise-floor replicates ran 10839 to 11036, the masking
control `t1c-mlm15-m80` is read at 10500, and **no run has ever been scored
downstream at 5000 steps**, so there is no sigma there for the metric that
decides this. A 5000-step screen could therefore only be read on eval loss,
which is exactly the measure this ablation expects to barely move. And it would
not have saved anything: six screens plus six resumes to 10500 costs what six
direct runs cost, so the screen only pays if it lets arms be dropped, on the one
signal least likely to separate them.

`max_wallclock_hours` is raised to 7.0 for these arms so that `max_steps` binds
rather than the clock: at the measured 1.98 s/step, 10500 steps is 5.78 h and
the standing 6 h limit leaves only ~14 min of margin, which one slow node would
eat. A fixed-step comparison wants the step count to be the thing that stops it.

They are read against `t2-mod-long` at equal step 10500. That run only has to
*reach* 10500, not finish, so none of this waits on it.

One asymmetry to state: `t2-mod-long` carries the 96-step data overlap from the
unpatched pinned tree and these arms do not, because the fix is now in. Below
resolution, logged in
[findings.md](findings.md#terminal-checkpoints-did-not-record-the-data-position).

### Result: not adopted

Ran 2026-09-12 as jobs 1763082-1763087. **No arm is worth adopting and the base
theta stands.** Supervised contact and PGYM both decline smoothly as theta
falls (Spearman 0.900 against log M each), loss is blind to it, and the
pre-registered decider turned out to be the one readout with sixteen times the
zig-zag of the others. Full table and reasoning in
[findings.md](findings.md#rope-theta-a-clean-negative-and-the-jacobian-readout-is-the-unstable-one).

### Deciding metric

Long P@L on `selected_protein`, per the [standing rule](method.md#decision-rule),
with loss diagnostic. Theta is a positional knob, so loss may barely move while
long-range contact does; supervised contact is a second read now that it runs.

Pre-registered caveat: training sequences are <=512 but the contact sets are
not, so low-theta arms could lose on extrapolation rather than representation.
Measured on the dev-mode subsets that actually get scored: `selected_protein`
2/71 over 512 (max 592, 1.16x), `casp14` 0/38, `casp15` 4/38 (max 573). The
deciding metric is therefore clean and the >512 stratum is a footnote; casp15
gets reported both ways.

Struck from the original eight: rmsnorm, swiglu and QK norm are now in the base
(see [decisions.md](decisions.md#the-modernized-baseline-rmsnorm-swiglu-qk-norm)),
QK-norm ordering (`reorder_RoPE_QKNorm`) goes with it, GQA is dropped, and tied
embeddings and span masking were already struck.

### All-global: result

Ran 2026-09-12 as job 1766751, fixed-step 10500 after the trace showed the arm is
compute-neutral. **Not adopted**: tied on the deciding metric (-0.0003) and worse
on all eight others, while costing more wall clock. Tier 2a is therefore closed
with the base unchanged on both knobs. Table and reasoning in
[findings.md](findings.md#all-global-attention-worse-or-tied-on-everything-and-not-adopted).

**At 13k and after the decay (2026-10-07):** all seven arms continued to 13000
and decayed to 15000. Same verdict: nothing beats B, E loses everywhere.
[results.md](results.md#position-encoding-at-13k-and-after-the-decay-2026-10-06).

## Tier 2b: MoE and canon layers

**MoE**, matched active parameters, swiglu (now the base, and also forced:
`use_moe=true` refuses geglu, `config.py:432`). **There is always a shared
expert**: `MoELayer` builds one `ModernBertSwiGLUMLP(config)` that processes
every token (`moe.py:468`) at the same `intermediate_size` as a routed expert,
with no knob to disable it. So active MLP width is
`(moe_top_k + 1) x intermediate_size` and sparsity is
`(moe_num_experts + 1) / (moe_top_k + 1)`. Any grid that ignores the shared
expert is wrong on both axes.

`moe_leading_dense_layers: 2`, so 30 of 32 layers are MoE. The two dense layers
are built at `moe_dense_intermediate_size = (top_k + 1) * intermediate_size`
(`modeling.py:959`), which is 2688 in every cell, so they are exactly base MLPs
and active width is matched in every layer of the model.

| cell | active | `moe_top_k` | expert inter | sparsity | `moe_num_experts` | total |
|---|---|---|---|---|---|---|
| g4-S8 | 4 | 3 | 672 | 8x | 31 | 2.135B |
| g4-S12 | 4 | 3 | 672 | 12x | 47 | 3.126B |
| g7-S8 | 7 | 6 | 384 | 8x | 55 | 2.135B |
| g7-S12 | 7 | 6 | 384 | 12x | 83 | 3.127B |

Active MLP params per layer are 8,257,536 in every cell, same as dense. Measured
active parameters per token: 402.19M against the dense base's 399.6M, the
difference being the routers (30 x 1024 x E), which is unavoidable.

**Why 7x384 and not 8x336.** Expert width turns out not to predict step time at
all; `top_k` does, monotonically, because dispatch and combine traffic scales
with token replication. 8 active at inter 336 costs +8.5% over 4 active at 672,
where 7 active at 384 costs +2.1%, and both match active parameters exactly. See
[findings.md](findings.md#expert-width-does-not-matter-top_k-does). `inter 512`
at top_k 4 was considered and rejected: it is the only candidate that misses
exact active matching (-4.76%) and it is still slower per active FLOP than 672.

**`micro_batch_seqs: 64`, uniformly, forced by memory.** At 128 the g7-S12 cell
OOMs: the compiled graph materializes `(T*top_k, *)` buffers that sonicmoe's
fusion avoids in eager, 1.6 GB per layer at `T*top_k = 65536*6`. 64 fits, at
97.9% of the card on the worst cell. grad_accum goes 4 -> 8. This is applied to
all four cells so the within-grid comparison is unaffected, and under wall-clock
matching the MoE arms pay for the extra micro-steps themselves. See
[findings.md](findings.md#moe-ooms-at-micro_batch_seqs-128-and-an-eager-per-layer-probe-could-not-see-it).

**Backend `sonicmoe`**, confirmed in the compiled graph (`quack.gemm_gated_out`,
`quack.gemm_out`, `sonicmoe._router_forward`), not a silent cutlass fallback.
No CUTLASS prebuild is needed on this path.

**Submit MoE arms ungated.** `tools/trace_gate.py` cancelled a healthy MoE smoke
and blacklisted four innocent nodes; its threshold assumes constant parameter
count and does not transfer at 3.13B. Recalibrating it to fit one run is the
post-hoc tuning this document exists to prevent. See
[findings.md](findings.md#the-trace-gates-threshold-does-not-transfer-to-moe).

### LR

MoE gets its own sweep: 2e-2 was tuned on dense, and the router's gradients and
the load-balance loss have no claim on transferring. Fixed-step `max_steps:
5000`, the same length as the dense re-sweep, on `g4-S12`, which is jul30's
winning shape and the fastest cell:

| arm | LR |
|---|---|
| `t2b-moelr-1e-2` | 1e-2 |
| `t2b-moelr-1.41e-2` | 1.41e-2 |
| `t2b-moelr-2e-2` | 2e-2 (the dense pick) |
| `t2b-moelr-2.83e-2` | 2.83e-2 |

Launched 2026-09-13 as jobs 1773336-1773339. The standing edge rule applies: if
the winner lands at 1e-2 or 2.83e-2, extend the grid before building on it.

Then a **2-arm transfer check on `g7-S12`** at the g4 winner and its higher
neighbour, because top_k 3 against top_k 6 changes the router's gradient
statistics and there is no argument that one LR serves both. Sparsity is assumed
not to move the LR (same active width, same top_k); that assumption is untested
and is recorded here as such.

### Budget

At the smoke's measured 2714 ms/step for g7-S12 (38.4% MFU, +35% over the dense
base's ~2010 ms), a 6.5 h arm reaches ~8600 steps against dense's ~11600. The
four grid arms are wall-clock matched at 6.5 h against a 7 h allocation.

**Canon** at kernel size 7 (the only size with CuTe backward kernels and the one
the perf work targeted), sweeping `canon_layers_mode` x `canon_layer_set`.
rmsnorm in the base retires the Triton-fallback handicap that prerequisite P7
flagged, so these arms now run on the CuTe path they were designed for.

## Tier 2c: our own ideas (24 short + 8 long)

**Done 2026-10-14, nothing adopted.** Ran 6 of the 8 arms (no Loopie, no
Huginn) on dense base v2: 24 sweep runs at 5000 steps (18 + 6 edge
extensions), then the top two LRs per arm resumed to round(13000 / f) and
decayed round(2000 / f). Results and verdict in
[results.md](results.md#tier-2c-our-own-ideas-on-dense-base-v2-2026-10-06-to-10-14).
The original design follows.

Each arm gets a short fixed-step LR sweep, then one longer wall-clock-matched
run at its best LR.

| id | change | key |
|---|---|---|
| C1+C2 | residual lambdas + x0 lambdas, combined into one arm | `use_resid_lambdas` + `use_x0_lambdas` |
| C3 | ProRes | `use_prores` |
| C4 | paired-head attention | `use_paired_head_attention` |
| C5a | mHC-lite, layer boundary | `use_mhc_lite` + `mhc_lite_wrapping_level: layer` |
| C5b | mHC-lite, sublayers | `use_mhc_lite` + `mhc_lite_wrapping_level: sublayers` |
| C6 | NOBLE | `use_noble` |
| C7 | Loopie | `use_loopie` |
| C8 | Huginn recycling | `use_huginn_looping` |

Struck: RePO. Confirmed 2026-09-11: 8 arms, so 24 short runs and 8 long ones.
The meeting's 27 and 9 predated merging x0 into the resid-lambda arm.

C1+C2 are unblocked: the lambdas are fp32 parameters cast to bf16 for the
forward with fp32 masters in the optimizer, the ordinary mixed-precision path,
not the `cast_forward_inputs` problem that broke the RoPE angles. Values near
1.0 have ~0.4% spacing in bf16 and updates accumulate in the fp32 master.
Checked under a real 1-rank FSDP2 with `MixedPrecisionPolicy`.

## Eval work the new metrics need

The team's reported set is PGYM Spearman total, zero-shot contact, supervised
contact, NewPISCES365, and subcell. Only NewPISCES365 has no path today.

| metric | status |
|---|---|
| PGYM Spearman (total) | works |
| contact, zero-shot | works |
| contact, supervised | **works, as of 2026-09-11.** The eager-attention recompute was the cheap route and it landed. `autoeval_supervised_contact.py` calls `embedder.compute_attention_map(sequence)`, which no kernel can answer because both FA3 and SDPA fuse the softmax away, so the probabilities are recomputed inside `ModernBertAttention` on the padded SDPA path the eval embedder already uses. That location is what makes it cheap: q and k there already carry RoPE and QK norm, and the mask passed to the kernel already encodes the sliding window, so none of it is rebuilt from config. Verified against the kernel itself: `probs @ v` matches SDPA to 1e-5 on CPU on both a global and a sliding layer, and on GPU under bf16 autocast to 2e-2, which is the kernel's own output rounding rather than the recompute's error. That comparison is too coarse to see whether the softmax is accumulated in fp32, so the fp32 accumulation is pinned separately by an invariance test (same q/k in, same probabilities out, inside an autocast region or not) that fails without the guard. 18 min per checkpoint in dev mode. Scores land beside zero-shot on all three sets, e.g. `selected_protein` long P@L2 0.529 supervised against 0.513 zero-shot on `t0-rep1` |
| subcell (`scl`) | **available on the upstream-synced eval branch.** It is a `sequence_to_class` task in `PBC_SUPERVISED`, which upstream re-enabled in `d3c5e79` along with a custom embedder exposing per-residue and per-sequence embeddings. The pinned tree still has it in `REMOVED_FRAMEWORKS`; the synced branch does not. Not free at runtime though: the framework trains a task head per task per checkpoint, so it needs timing before it becomes a column |
| newPISCES364 | **works, as of 2026-09-12: accuracy 0.8032** on the new base. It is a test split of PBC's **secondary structure** task, not a contact set, which is why it never turned up next to the contact data: `SET=newPISCES364` on 364 of the 11205 sequences in `secondary_structure.fasta`, alongside casp12/13/14 and no `SET=test` at all. Biotrainer needs no change to report it (`get_split_lists` has `case _: # Treat all other sets as testing sets`, and `testing_ids` is a dict), and it arrives through `pbc_supervised` together with the rest of the supervised menu. An earlier revision of this row claimed it did not exist anywhere; that was a search for a dataset *name* when it is a `SET=` value inside a FASTA, in a framework this series had never run. See [findings.md](findings.md#newpisces364-and-why-pbc_supervised-had-never-run) |

Supervised contact was worth more than its place in the priority list suggested,
which is why it went first: it is the one measurement that can settle the 100/0/0
masking question, because it reads a probe on frozen features instead of the
model's own output sensitivity. It is now available to do that.

## Tier 3: greedy ladder (~16 runs)

Take the Tier 2 winners and add them one at a time in descending effect size,
2 seeds per rung. A rung stays only if it beats the previous rung by more than
2 x sigma_seed. This is where interactions show up: norm, activation and
optimizer are the three most likely to interact, and one-at-a-time cannot see
that.

## Struck: Tier 4 and Tier 5

Leave-one-out (12 runs) and the 600M transfer plus fp8 pair (6 runs) are out of
scope by team decision, 2026-09-11. Both consequences are logged in
[method.md](method.md#overrides-logged): the final recipe is not leave-one-out
validated, and nothing is checked at a second scale. If any budget comes back,
the 4 transfer runs buy more than the 2 fp8 runs, because the batch size was
chosen at 400M and its critical batch moves with scale.

## The decay campaign

**Superseded 2026-10-05** by the WSD protocol at the top of this file: every
arm decays 2000 steps (or 2000 / f) from 13000, 1-sqrt to 0.1, continuing the
data position. The original design follows.

The ladder runs `lr_schedule: warmup_stable` only. Decay is a separate campaign
later, `resume.mode: decay` from the saved checkpoints, annealing to
`lr_decay_to_fraction`. Deliberately not chained with `--dependency=afterok`: a
stable job that dies would take its decay job with it silently, and running it
separately lets us pick the decay length after seeing the stable results.

Three things to get right when it is set up:

- `decay_steps` at 8.33% of the steps the stable phase actually completed gives
  every arm the same 30 min of decay in wall clock, whatever its speed.
- `decay_shape: linear` (set in the base, inert during the stable phase).
- **Round `decay_steps` to a multiple of `eval_steps`.** Eval fires on
  `global_step % eval_steps == 0 or at_wsd_stable_end` and there is no eval after
  the loop, so an unrounded `decay_steps` ends the run with no score.

30 min is about 8% of a run, against the usual WSD practice of 10 to 20%. Worth
revisiting now that it is no longer locked to the stable runs.

## Side ablation: Ling-style routing on g7-S12 (2026-10-30)

Our router is a hybrid: Ling-V2 machinery (sigmoid scores, fp32 router,
zero-mean expert bias, z-loss) bolted onto the defaults of the first MoE commit
(proportional bias step, soft aux loss 0.01, z-loss 5e-5, routed scale 1.0).
These arms ask whether the actual Ling-mini-2.0 / DeepSeek-V3 recipe does
better, and what the routed scale should be.

Tree v4 = tree v3 + `moe_bias_update_mode` (`pkgs/nanoplm-v4.patch`). The
default, `proportional`, is bit-identical to v3. `sign` is Ling-V2's
`--bias-zero-mean-update`: every expert's bias moves by a fixed +-1e-3 toward
balance, then the vector is recentered.

All arms are g7-S12 (83 experts, top-6, 1 shared, width 384), muon LR 1.41e-2,
5000 fixed steps, against `lr3-g7-S12-1.41e-2` (job 2576789, eval 2.1866 @5000).
Configs derive from `configs/ling-base-g7-S12.yaml` (the baseline with the
v4 defaults spelled out).

| arm | bias step | aux loss | z-loss | routed scale | job |
|---|---|---|---|---|---|
| baseline | proportional | 0.01 | 5e-5 | 1.0 | 2576789 |
| ling-g7-S12-s1 | sign | 0 | 3.5e-6 | 1.0 | 2596677 |
| ling-g7-S12-s2.5 | sign | 0 | 3.5e-6 | 2.5 (Ling, DSv3) | 2596678 |
| ling-g7-S12-s6 | sign | 0 | 3.5e-6 | 6 (= top-k) | 2596679 |

`s1` vs baseline isolates the balancing recipe; `s1 -> s2.5 -> s6` is the
routed-scale sweep on top of it. Scale 6 makes the routed branch's combine
weights sum to 6 rather than 1, so each routed expert enters the residual at
the same weight as the shared expert.

Not copied from Ling: group-limited routing (`n_group 8 / topk_group 4` exists
for node-limited dispatch, and 83 experts do not split into equal groups) and
`first_k_dense 1` (we keep 2 leading dense layers, so the compute match holds).

**Decision rule, fixed before results.** The 3 dense base2 seeds span 1.1e-3
at step 5000. An arm counts as better than baseline only if it is lower by
more than 2e-3. A win at 5000 steps then needs a full ovh5-budget run with
decay before it is adopted. Caveat: the muon LR was tuned at scale 1, and the
routed scale changes the effective step size of the routed experts, so a loss
at scale 6 may be an LR mismatch rather than a bad scale.

**LR sweep per scale (added 2026-10-30, at the user's request).** Each scale
also runs at muon LR 7.07e-3, 1e-2 and 2e-2 (the 1.41e-2 point is the arm
above), so scales are compared at their own best LR:

| scale | 7.07e-3 | 1e-2 | 1.41e-2 | 2e-2 |
|---|---|---|---|---|
| 1 | 2596782 | 2596783 | 2596677 | 2596784 |
| 2.5 | 2596785 | 2596786 | 2596678 | 2596787 |
| 6 | 2596788 | 2596789 | 2596679 | 2596790 |

Configs are `ling-g7-S12-s<scale>-lr<lr>`. If a scale's best point is at
either edge of its grid, it gets one more point one sqrt(2) step beyond it.

## Budget

Pre-roadmap budget, kept for the record; the roadmap above is current.

| item | runs | note |
|---|---|---|
| LR re-sweep on the new base | 7 | fixed-step 5000, blocks everything below |
| modernization long run | 1 | vs stock ModernBERT, control already exists |
| ~~1c seeds~~ | 0 | struck: split closed at 80/10/10 |
| 2a | 6 | 2 arms x 3 LRs |
| 2b | ~8 | MoE sparsity x granularity, canon mode x set at K=7 |
| 2c short | 24 | 8 arms x 3 LRs, fixed-step 5000 |
| 2c long | 8 | one wall-clock run per arm at its best LR |
| 3 | ~16 | greedy ladder |
| **left** | **~71** | about 6,800 GPU-hours |

Roughly half the original ~140, mostly from striking Tier 4 and 5 and from
folding six Tier 2a arms into the base. The tiers are ordered so cutting from
the bottom of 2c costs the least.

## Still open

- **Decide what, if anything, of Tier 2c is re-checked on the MoE base.**
  Paired-head was the only loss win (-0.0024, one seed, downstream tied) and is
  not going into the base.
- **mHC layer's LR grid looked off** (1e-2 far behind, flat from 4e-2 up). The
  DeepSeek-V4 init may suit sublayer wrapping rather than layer wrapping; worth
  a look before any mHC re-run.
- **The eval tree must track the training tree.** Any model change has to be
  mirrored in `eval/nanoplm-src-sync` before its checkpoints are scored
  ([findings](findings.md#the-eval-tree-is-its-own-tree-and-lags-the-training-tree)).

- **Development-mode aggregation is now the standard**, so the earlier note
  that a full-mode run was owed is retired. What is owed instead is a
  full-mode rescore of whatever the paper reports as headline numbers, since
  dev mode is 1.7x blunter and subsamples the benchmark.
- **Stability metrics need a definition.** Everything needed is already logged
  (434 keys per step in `debug_layerwise.jsonl`: per-family grad norms, weight
  norms, per-layer residual RMS, logits abs_max/entropy/norm, plus per-step
  `grad_norm` in the Slurm log). A reporting set has to be picked and a tool
  written to extract it per arm.
- **The loss / long-range-contact decoupling** in
  [results.md](results.md#one-arm-where-loss-and-structure-disagree). Val loss
  being blind to a downstream collapse is worth understanding before the final
  runs.
- **P3**: resid-lambda tensors reach the layer as forward args, so FSDP casts
  them to bf16. Deferred and accepted; revisit if the resid-lambda arm behaves
  oddly.
- **P7**: canon layers on a layernorm base hit the slow Triton fallback.
  Accepted; the writeup notes the kernel handicap, so a loss there is not
  evidence against the idea.
- **P8**: `9bed380` removed the TE backend and Tier 5 promises an fp8 pair.
  Verify when Tier 5 is reached.
- **~630 GB of optimizer state is prunable.** Checkpoints are 4.5 GB (model
  1.6 GB + optimizer 3.2 GB) against 39 TB free, so not a constraint, but only
  `pytorch_model.bin` is needed for biotrainer.
- **The final long run must re-check the top two LRs with decay attached.** A
  fixed-step sweep picks under a constant LR and is biased low.

## Considered, not included

Say the word and any of these come in, 2 to 5 runs each.

- sequence length 1024 instead of 512
- warmup length (currently 250 steps at 4.19M tokens)
- decay length and shape (currently 30 min, linear)

**Full runs for s1 and s2.5 (added 2026-10-31, at the user's request).** Both
continue from their own 5000-step checkpoint (`resume.mode: continue`) to the
moe3-g7-S12 trunk budget, 9723 steps, then decay 1496 steps to 11219, the same
as `moe3-g7-S12` + `moe3-g7-S12-decay` (2587908/09). Each at its own best LR from
the sweep: s1 at 1.41e-2 (from 2596677), s2.5 at 1e-2 (from 2596786). Tree v4.
A continue resume saves into the resumed run's dir, so each 5000 checkpoint was
copied to `checkpoint-stable-end-s5000` first; the decay chain checks the
trunk's stable-end is at step 9723 before decaying.

| arm | trunk | decay | eval |
|---|---|---|---|
| ling-g7-S12-s1-long | 2604856 | 2604858 | 2604860 |
| ling-g7-S12-s2.5-lr1e-2-long | 2604857 | 2604859 | 2604860 |
| baseline moe3-g7-S12 | 2587908 (done, 2.1123 @9723) | 2587909 | 2604861 |

Eval: dev battery (PGYM, both contact readouts, PBC supervised), bt211 stack,
at s9723 and d11219, eval source = tree v4 (strict-loads v4 checkpoints).
Compare eval loss at 9723 and after decay, and downstream, against the baseline.
