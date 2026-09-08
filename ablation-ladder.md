# Ablation ladder

For review. Nothing runs until this is signed off.

Every run: 6 h stable + 30 min decay, 4 nodes, 16 GH200. About 104 GPU-hours.
Run shape is two chained Slurm jobs (see "How a run works").

Deciding metric: **biotrainer PBC and PGym**, run later on the stored
checkpoints. Biotrainer is not ready yet, so the ablations run now and store
every checkpoint for scoring afterwards.

Post-decay eval loss is a secondary signal, not the decider. It is still worth
getting right (see P1/P2, both now fixed) because it is the only signal we have
while the runs are in flight.

Two consequences:

- **Every checkpoint has to survive.** ~135 runs at 399.6M params is on the
  order of a terabyte. Check the fscratch quota and the real per-run checkpoint
  size before Tier 0, not at run 60.
- **Arms cannot be ranked until biotrainer lands.** The tier structure below
  still holds, but Tier 3 onward needs the real metric, so plan to run Tiers 0
  to 2 first and hold.

---

## Prerequisites (must land before Tier 0)

These are not optional. Both break the deciding metric.

| id | problem | fix |
|---|---|---|
| P1 | DONE (`4e3bbfc`). Eval inherited the whole training masking recipe, not just the rate: `mlm_probability`, the 80/10/10 split, and the strategy. So A11-A14 all changed the eval task too. | Added `eval_mlm_probability`, `eval_mask_replace_prob`, `eval_random_token_prob`, `eval_keep_probability`, `eval_mlm_masking_strategy`. Pinned in the base config at 15% token, 80/10/10. |
| P2 | DONE (`4e3bbfc`). Eval masks were redrawn every call from global RNG. | Added `eval_mask_seed`. The mask is seeded from a hash of the batch's token ids, so it is stable across arms, runs, worker counts and batch order. |
| P3 | Resid-lambda tensors reach the layer as forward args, so FSDP casts them to bf16. Resid lambdas is an arm. | Confirm the dtype. Move to buffers if it matters. |
| P4 | Launcher has no `--time` default (same footgun as `--nodes` was). If the stable job hits the Slurm limit instead of its own wall-clock budget it exits non-zero, and `afterok` never fires the decay job. | `#SBATCH --time=07:00:00` for the stable job; the decay job gets its own. |
| P5 | `resume.mode: decay` has never run in this series. | End-to-end chain smoke: short stable job, `--dependency=afterok`, decay job. Confirm `training_state.json` carries `global_step`, that the decay config leaves `max_wallclock_hours` unset so it is step-bounded, and that a final eval actually fires. |
| P6 | **Eval costs 4.0 s and fires every 250 steps.** Over a 6 h run at ~511 ms/step that is ~169 evals, about 11 min, roughly 3% of the budget. | Confirmed by measurement, not estimate. Decide whether to keep `eval_steps: 250` or widen it. Your call; it is real compute. |
| P7 | Canon layers (C2) on a layernorm base hit the slow Triton fallback. The CuTe backend covers rms_conv and bare conv only. | C2 would lose on wall-clock because of a kernel gap, not because the idea is bad. Either run it with bare conv, or defer canon to Tier 3 conditional on rmsnorm (A1) winning. |
| P8 | `9bed380` removed the TE backend. `fp8` still exists as its own module, but Tier 5 promises an fp8 pair. | Smoke `fp8: true` on the pinned tree before promising it. |

---

## How a run works

The pipeline does not do 6 h + decay in one job. It is two:

1. **Stable job.** `lr_schedule: warmup_stable`, `max_wallclock_hours: 6.0`.
   Stops on the clock and writes `checkpoint-stable-end`.
2. **Decay job.** `resume.mode: decay`, annealing from the stable LR down to
   `lr_decay_to_fraction`. Chained with `--dependency=afterok`.

`decay_steps` is set to 8.33% of the steps the stable phase actually completed,
so a faster arm gets a proportionally longer decay in steps and the same 30 min
in wall clock.

**Round `decay_steps` to a multiple of `eval_steps`.** Eval fires on
`global_step % eval_steps == 0 or at_wsd_stable_end`, and there is no eval after
the loop. If `decay_steps` is not a multiple of 250 the final decay step gets no
eval at all, and the run ends with no score. Rounding avoids a code change.

One note: 30 min is about 8% of the run. Common WSD practice is 10 to 20%. Not
changing it, just flagging that it is a lever.

---

## Tier order note

Tier 1 runs **before** Tier 0. The noise floor has to be measured at the LR the
base actually uses, otherwise sigma is measured at one LR and every arm is
compared against a base at another.

## Tier 0 - noise floor (6 runs, after Tier 1)

No arm is judged before this exists.

| runs | what | gives |
|---|---|---|
| 3 | base, 3 different seeds | sigma_seed |
| 3 | base, same seed, 3 repeats | sigma_repeat |

sigma_repeat is not zero. Non-deterministic kernels (atomics in reductions,
autotuned kernel choice, NCCL reduction order) mean identical inputs do not
give identical loss, and it compounds.

**Decision rule, fixed now.** Compare the arm's best-over-LR result against the
base's best-over-LR result from Tier 1. An arm wins only if it is better by more
than 2 x sigma_seed. Inside that band it is reported as "no measured effect".

Best-over-LR is optimistic on both sides, which is why the base gets the same
treatment. Naming the rule now is the whole point of pre-registering it.

---

## Tier 1 - base LR sweep (5 runs)

5 LRs on the base, 2x spacing, centred on 1e-4.

Why this is not optional: a single LR tuned on the base is unfair to any arm
that moves the optimum. Without it, an optimizer arm that loses may have lost
only because it ran at AdamW's LR. This is what makes Tier 2 interpretable.

---

## Tier 2 - one factor at a time (about 75 runs)

Each arm changes exactly one thing from the base. No arm builds on another.
Each arm runs at **3 LRs** (0.5x / 1x / 2x its expected optimum), 1 seed. The
LR curve is the robustness check: an arm that only wins at one LR did not win.

An arm that changes speed gets more or fewer tokens in its 6 h. That is the
design, not a confound.

### 2a. Standard recipe knobs

| id | change | key | why |
|---|---|---|---|
| A1 | rmsnorm | `norm_type` | cheaper, widely adopted since ModernBERT |
| A2 | swiglu | `mlp_activation` | the common alternative to geglu |
| A3 | squared relu | `mlp_activation` | cheap, competitive in recent work |
| A4 | srelu | `mlp_activation` | ours, sparse activations |
| A5 | QK norm on | `use_qk_norm` | attention-logit stability at depth |
| A6 | QK norm before RoPE | `reorder_RoPE_QKNorm` | order is not obviously settled |
| A7 | all-global attention | `attn_layer_pattern` | is alternating local/global earning its place |
| A8 | rope theta 10k | `global_rope_theta` | 160k is inherited, not tuned for 512-token proteins |
| A9 | untied embeddings | `tie_word_embeddings` | costs params, may buy output quality |
| A10 | GQA, 8 kv heads | `num_kv_heads` | cheaper attention, more tokens in 6 h |
| A11 | MLM 15% | `mlm_probability` | 30% is high; needs P1 to be interpretable |
| A12 | MLM 40% | `mlm_probability` | the other direction |
| A13 | span masking | `mlm_masking_strategy` | BERT-style spans vs per-token |
| A14 | mask 100%, no 80/10/10 | `mask_replace_prob` | the 10/10 split is cargo-culted from BERT |
| A15 | weight decay 0.1 | `adam_weight_decay` | 1e-5 is very low |
| A16 | beta2 0.95 | `adam_beta2` | 0.98 is inherited |

### 2b. Optimizers

| id | change | why |
|---|---|---|
| B1 | muon | matrix-aware, the current default in a lot of frontier work |
| B2 | normuon | our best jul30 result, and the only one that cleared noise |
| B3 | nordion2 | newly wired up |
| B4 | stable_adamw | cheap robustness check on the AdamW baseline |

These cannot use "0.5x/1x/2x of the base optimum": Muon-family LRs live on a
different scale from AdamW's entirely. Each gets its own 5-point sweep anchored
on its own published default. That is +8 runs beyond the 3-LR rule, counted in
the budget below.

### 2c. Our own ideas

These need a second seed before anyone should believe them. That is +10 runs at
each arm's best LR, counted in the budget.

| id | change | key |
|---|---|---|
| C1 | residual lambdas | `use_resid_lambdas` (see P3) |
| C2 | canon layers | `use_canon_layers` |
| C3 | RePO | `use_repo` |
| C4 | ProRes | `use_prores` |
| C5 | paired-head attention | `use_paired_head_attention` |
| C6 | mHC-lite | `use_mhc_lite` |
| C7 | NOBLE | `use_noble` |
| C8 | Loopie | `use_loopie` |
| C9 | Huginn recycling | `use_huginn_looping` |
| C10 | MoE | `use_moe` (two known issues: quack GEMM rejects bf16 so eval falls back to cutlass, and the CUTLASS grouped-GEMM JIT has a multi-rank build race, so prebuild the .so before srun) |

MoE is not parameter-matched to the base by construction. Decide before running
whether it is compared at matched active params or matched total params, and
say which in the paper.

---

## Tier 3 - greedy ladder (about 16 runs)

Take the Tier 2 winners, add them one at a time in descending effect size,
2 seeds per rung. A rung stays only if it beats the previous rung by more than
2 x sigma_seed.

This is where interactions show up. Norm, activation, and optimizer are the
three most likely to interact, and one-at-a-time cannot see that.

---

## Tier 4 - leave-one-out (about 12 runs)

For each ingredient in the final recipe, run the recipe without it, 2 seeds.

An ingredient that does not hurt when removed does not go in the paper's
recipe, however well it did alone. This is what catches ingredients that were
only ever measuring each other.

---

## Tier 5 - transfer and precision (6 runs)

| runs | what |
|---|---|
| 4 | base vs final recipe at 574.8M (h1152/L36), 2 seeds each |
| 2 | fp8 vs bf16 on the final recipe |

A win that does not survive the scale-up does not go in the paper.

---

## Budget

| tier | runs | note |
|---|---|---|
| 1 | 5 | base LR sweep, runs first |
| 0 | 6 | noise floor at the chosen LR |
| 2a | 48 | 16 arms x 3 LRs |
| 2b | 20 | 4 optimizers x 5 LRs |
| 2c | 30 | 10 arms x 3 LRs |
| 2c seeds | 10 | second seed on our own ideas |
| 3 | ~16 | greedy ladder |
| 4 | ~12 | leave-one-out |
| 5 | 6 | transfer + fp8 |
| **total** | **~153** | **~16,000 GPU-hours** |

Strike rows if that is too many. The tiers are ordered so cutting from the
bottom of 2c costs the least.

## Considered, not included

Say the word and any of these come in. Each is 2 to 5 runs.

- sequence length 1024 instead of 512
- global batch size (currently 1M tokens)
- warmup length (currently 1000 steps)
- decay length and shape (currently 30 min, 1-sqrt). We flagged 30 min as short
  versus the usual 10 to 20%, but did not test it.
