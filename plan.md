# Plan

What is left to run. Done work is in [results.md](results.md); the matching and
threshold rules are in [method.md](method.md).

Every architecture arm: 6 h stable phase, 4 nodes, 16 GH200, 4.19M tokens/step
(grad_accum 4), ~10,568 steps, ~44.3B tokens, about 96 GPU-hours. Compute-neutral
arms run fixed-step instead. Each arm changes exactly one thing from the base and
runs at **3 LRs** (0.5x / 1x / 2x its expected optimum), 1 seed. The LR curve is
the robustness check: an arm that only wins at one LR did not win.

## Now: the modernized base

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
1761232-1761238: 3.5e-3, 5e-3, 7e-3, 1e-2, 1.41e-2, 2e-2, 2.8e-2. Every point
shares one horizon, so the table needs no cross-horizon caveat, and the three
3500-step results above stand as an independent early read of the same curve. If
2.8e-2 wins, extend again.

This is a real change from the stock base, where 7e-3 was the interior optimum
three times over and 1e-2 was already +0.0050 worse. Expected direction: QK norm
bounds attention logits, which is what usually limits the learning rate. The base changed twice at once: rmsnorm + swiglu +
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

## Next: settle the masking split

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

## Tier 2a: the two remaining recipe knobs (6 runs)

Off the modernized base, 3 LRs each, wall-clock matched.

| id | change | key | why |
|---|---|---|---|
| A6 | all-global attention | `attn_layer_pattern` | is alternating local/global earning its place |
| A7 | rope theta 10k | `global_rope_theta` | 160k is inherited, not tuned for 512-token proteins |

Struck from the original eight: rmsnorm, swiglu and QK norm are now in the base
(see [decisions.md](decisions.md#the-modernized-baseline-rmsnorm-swiglu-qk-norm)),
QK-norm ordering (`reorder_RoPE_QKNorm`) goes with it, GQA is dropped, and tied
embeddings and span masking were already struck.

## Tier 2b: MoE and canon layers (~8 runs)

**MoE**, matched active parameters, swiglu (now the base, and also forced:
`use_moe=true` refuses geglu, `config.py:432`). **There is always a shared
expert**: `MoELayer` builds one `ModernBertSwiGLUMLP(config)` that processes
every token (`moe.py:468`) at the same `intermediate_size` as a routed expert,
with no knob to disable it. So active MLP width is
`(moe_top_k + 1) x intermediate_size` and sparsity is
`(moe_num_experts + 1) / (moe_top_k + 1)`. That is why jul30's 48 routed experts
at top_k 3 were 49 total and 12.25x, not 16x. Any grid that ignores the shared
expert is wrong on both axes.

| cell | active experts | `moe_top_k` | expert `intermediate_size` | sparsity | `moe_num_experts` | total params |
|---|---|---|---|---|---|---|
| g4-S8 | 4 | 3 | 672 | 8x | 31 | 2.13B |
| g4-S12 | 4 | 3 | 672 | 12x | 47 | 3.12B |
| g8-S8 | 8 | 7 | 336 | 8x | 63 | 2.13B |
| g8-S12 | 8 | 7 | 336 | 12x | 95 | 3.12B |

Dense base is 0.40B for reference. Active MLP params per layer are 8,257,536 in
every cell, same as dense.

**Why 8x is still in the grid even though jul30 landed on 12x.** The jul30
sparsity ablation (h1024/L19, inter 512, top_k 3, 3 h wall-clock matched):

| arm | total experts | sparsity | total params | final loss | MFU | tokens |
|---|---|---|---|---|---|---|
| moe4x | 13 | 3.25x | 0.41B | 2.2999 | 40.2% | 37.7B |
| moe8x | 33 | 8.25x | 0.95B | 2.2768 | 38.4% | 36.0B |
| moe12x | 49 | 12.25x | 1.37B | **2.2688** | 38.7% | 36.3B |

12x won, but the returns collapse: -0.0231 from 3.25x to 8.25x, then only
-0.0080 from 8.25x to 12.25x while total parameters grew 44%. And jul30 never
measured a noise floor, so -0.0080 may well be noise. Keeping 8x costs 2 runs
and re-tests that conclusion against a sigma we now have. If 12x wins again by
more than 2 sigma it is settled properly; if not, we ship a model 1B parameters
smaller for free.

Known issue: the CUTLASS grouped-GEMM JIT has a multi-rank build race, so
prebuild the `.so` before `srun`. The old warning that eval falls back to
cutlass because the quack GEMM rejects bf16 is stale: `38621fa` (2026-09-02)
casts `Wi`/`Wo` to bf16 at load and `moe.py` casts activations in and back out,
`_force_cutlass_moe_backend` is gone, so sonicmoe checkpoints evaluate on the
sonicmoe path.

**Canon layers** at **kernel size 7** (team decision, from the earlier kernel
benchmarks), sweeping `canon_layers_mode` x `canon_layer_set`. rmsnorm in the
base retires the Triton-fallback handicap that prerequisite P7 flagged, so these
arms now run on the CuTe path they were designed for.

## Tier 2c: our own ideas (24 short + 8 long)

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
contact, NewPISCES365, and subcell. Only the first two run today.

| metric | status |
|---|---|
| PGYM Spearman (total) | works |
| contact, zero-shot | works |
| contact, supervised | **needs code.** `autoeval_supervised_contact.py` calls `embedder.compute_attention_map(sequence)`, and nanoPLM runs FA3, which never materialises an attention matrix. Needs an eager-attention recompute in the eval wrapper, plus lifting `pbc_supervised` out of `REMOVED_FRAMEWORKS` |
| subcell (`scl`) | **needs code.** It is a `sequence_to_class` task in `PBC_SUPERVISED`, so it needs per-sequence embeddings out of the wrapper, and that framework is also in `REMOVED_FRAMEWORKS` |
| NewPISCES365 | **does not exist in biotrainer at all.** Searched the pinned PR #192 tree, `biotrainer-core`, all seven branches across the three forks (sacdallago upstream, peymanvahidi, alint77), every commit message in full history, and the downloaded datasets: no match for "pisces" anywhere. The supervised contact sets are train/val plus casp14, casp15 and selected_protein. It needs a data source and a task protocol from whoever proposed it |

Doing supervised contact first is worth more than its place in the priority list
suggests: it is the one measurement that could settle the 100/0/0 masking
question, because it reads a probe on frozen features instead of the model's own
output sensitivity.

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

## Budget

| item | runs | note |
|---|---|---|
| LR re-sweep on the new base | 7 | fixed-step 5000, blocks everything below |
| modernization long run | 1 | vs stock ModernBERT, control already exists |
| 1c seeds | 4 | settle the masking split, fixed-step at 10500 |
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
