# Ablation ladder

Status: Tier 1 submitted 2026-09-08 (10 jobs, ids in
`sep07_abl/run/tier1_jobids.txt`). Tier 2 onward is still open for review.

Every run: 6 h stable phase, 4 nodes, 16 GH200. About 96 GPU-hours. The 30 min
decay is a separate later campaign off the saved checkpoints, not chained
behind each job (see "How a run works").

Deciding metric: **biotrainer PBC and PGym**, run later on the stored
checkpoints. Biotrainer is not ready yet, so the ablations run now and store
every checkpoint for scoring afterwards.

Post-decay eval loss is a secondary signal, not the decider. It is still worth
getting right (see P1/P2, both now fixed) because it is the only signal we have
while the runs are in flight.

Two consequences:

- **Every checkpoint has to survive.** Measured at 4.5 GB per run (model
  1.6 GB + optimizer 3.2 GB), so about 630 GB for the series against 39 TB free
  on fscratch. Not a constraint.
- **Arms cannot be ranked until biotrainer lands.** The tier structure below
  still holds, but Tier 3 onward needs the real metric, so plan to run Tiers 0
  to 2 first and hold.

---

## Prerequisites

Status as of the last review. Only the open ones gate the start.

| id | problem | fix |
|---|---|---|
| P1 | DONE (`4e3bbfc`). Eval inherited the whole training masking recipe, not just the rate: `mlm_probability`, the 80/10/10 split, and the strategy. So A11-A14 all changed the eval task too. | Added `eval_mlm_probability`, `eval_mask_replace_prob`, `eval_random_token_prob`, `eval_keep_probability`, `eval_mlm_masking_strategy`. Pinned in the base config at 15% token, 80/10/10. |
| P2 | DONE (`4e3bbfc`). Eval masks were redrawn every call from global RNG. | Added `eval_mask_seed`. The mask is seeded from a hash of the batch's token ids, so it is stable across arms, runs, worker counts and batch order. |
| P3 | Resid-lambda tensors reach the layer as forward args, so FSDP casts them to bf16. | DEFERRED, accepted for now. Revisit if the resid-lambda arm behaves oddly. |
| P4 | Launcher had no `--time` default (same footgun as `--nodes` was). | DONE. `#SBATCH --time=07:00:00`. |
| P5 | `resume.mode: decay` has never run in this series. | NO LONGER BLOCKING. The decay runs are a separate campaign off the saved checkpoints, so this gets verified when that campaign is set up, not before the ladder starts. |
| P6 | Eval cost 4.0 s every 250 steps, about 11 min or 3% of a 6 h run. | DONE. `eval_steps: 500`, halving it to ~1.5%. Eval loss is only a secondary signal now, so the cadence does not need to be tight. |
| P7 | Canon layers (C2) on a layernorm base hit the slow Triton fallback. | ACCEPTED. C2 runs as-is. Note in the writeup that it carries a kernel handicap on a layernorm base, so a loss there is not evidence against the idea. |
| P8 | `9bed380` removed the TE backend; Tier 5 promises an fp8 pair. | ACCEPTED. Verify when Tier 5 is reached, not now. |

---

## How a run works

**Two phases, run as separate campaigns, not chained.**

1. **Now: the stable phase.** `lr_schedule: warmup_stable`,
   `max_wallclock_hours: 6.0`. This is what the whole ladder below runs.
2. **Later: the decay runs.** `resume.mode: decay` from those saved
   checkpoints, annealing to `lr_decay_to_fraction`.

**The wall-clock stop writes `checkpoint-<step>`, not `checkpoint-stable-end`.**
Verified on a real run: "Stopped on the wall-clock budget at step 920/100000;
skipping the terminal 'stable-end' checkpoint. Latest state is
checkpoint-920." The pipeline reserves the terminal role name for a run that
actually reaches the end of its LR schedule, which a wall-clock stop never
does.

This is not a problem. The checkpoint is complete (model 1.6 GB + optimizer
3.2 GB + scheduler + RNG + configs, 4.5 GB) and `ResumeConfig.checkpoint_dir`
is an explicit path, so the decay campaign just points at the step-named
directory. Worth knowing so nothing goes hunting for a name that will not be
there.

Storage: one checkpoint per run at 4.5 GB (`save_steps` is off), so about
630 GB for the series. Only `pytorch_model.bin` (1.6 GB) is needed for
biotrainer, so the optimizer state can be pruned later if space gets tight.

Decoupling them is deliberate. Chaining with `--dependency=afterok` means a
stable job that dies for any reason silently takes its decay job with it, and
we would not find out until we went looking. Running the decay campaign
separately also lets us decide the decay length after seeing the stable
results, and it fits how the checkpoints get scored: biotrainer runs on them
later anyway.

Slurm time limit is 7 h for a 6 h stable phase. The headroom covers compile,
dataset setup and the final checkpoint write.

Two things to get right when the decay campaign is set up:

- `decay_steps` at 8.33% of the steps the stable phase actually completed
  gives every arm the same 30 min of decay in wall clock, whatever its speed.
- `decay_shape: linear` (set in the base; inert during the stable phase).
- **Round `decay_steps` to a multiple of `eval_steps`.** Eval fires on
  `global_step % eval_steps == 0 or at_wsd_stable_end` and there is no eval
  after the loop, so an unrounded `decay_steps` ends the run with no score.

One note: 30 min is about 8% of the run. Common WSD practice is 10 to 20%.
Worth revisiting when the decay campaign is planned, since it is no longer
locked to the stable runs.

---

## Tier 0b - global batch size (6 runs, runs FIRST)

Batch size gates everything, because both optimizers' LR optima move with it.
An LR sweep at 1M is not transferable to 4M, so this runs before Tier 1.

**The infra argument is about scale-out headroom, not comms overhead.**

Global batch caps how many GPUs can be used before the per-GPU batch has to
shrink, and a smaller local batch means smaller GEMMs and lower utilisation.
At `micro_batch_seqs` 128 and 512-token sequences, each GPU takes 65,536 tokens:

| global batch | GPUs at full local batch | nodes |
|---|---|---|
| 1.0M | 16 | 4 |
| 2.1M | 32 | 8 |
| 4.2M | 64 | 16 |
| 8.4M | 128 | 32 |

So 1M tokens caps us at 4 nodes before local batch starts dropping. 4M buys 16
nodes at the same local batch. That is the argument for a bigger batch, and it
is about the eventual full-scale run, not about these ablations, which are
pinned at 4 nodes.

**Comms overhead is NOT the argument.** FSDP2 does skip the reduce-scatter on
accumulation micro-steps (`set_requires_gradient_sync(at_accum_boundary)`), but
exposed (non-overlapped) NCCL is only **5.2 ms of a 511 ms step, 1.0%**,
measured from the h1024/L32 4-node trace by subtracting the compute-kernel union
from the NCCL union. Upper-bound saving is 2.6 ms at ga=2 and 3.9 ms at ga=4,
under 1% either way. Anyone reaching for grad-accum to save collectives at this
scale is reaching for nothing.

**At fixed wall-clock the token count is the same either way** (44.3B in 6 h at
any of these batches), so the only question is whether the model learns as much
from 10.5k steps of 4M as from 42k steps of 1M. That is the critical-batch-size
question, and it is measurable.

The probe is **token-matched, not wall-clock matched**, so it isolates the batch
effect:

| run | batch | grad_accum | steps | tokens | warmup | AdamW LR | NorMuon LR |
|---|---|---|---|---|---|---|---|
| b1M | 1.05M | 1 | 9600 | 10.07B | 1000 | 1e-4 | 1e-2 |
| b2M | 2.10M | 2 | 4800 | 10.07B | 500 | 1.41e-4 | 1.41e-2 |
| b4M | 4.19M | 4 | 2400 | 10.07B | 250 | 2e-4 | 2e-2 |

Both optimizers, 6 runs, ~1.4 h each. LRs scale by sqrt(B) from the 1M anchor,
which is the standard rule and an approximation: it will be somewhat unfair to
whichever optimizer scales differently, and that is a known limitation of the
probe rather than a result. Warmup is held at a fixed 1.05B tokens, not a fixed
step count, or warmup would eat a quarter of the 4M run. Step counts are
multiples of `eval_steps` so each run ends on an eval.

**Decision rule.** If 4M is within noise of 1M at equal tokens, take 4M: it
costs nothing in learning and buys 4x the scale-out headroom for the full run.
If 4M is clearly worse, the critical batch is below 4M, so take 2M or stay at
1M and accept the node ceiling.

The learning question is the only one the probe answers. The scale-out headroom
is arithmetic, not something to measure.

**ESM C's batch size is unverified.** It may be 4M; we have no local copy of the
paper or code to check, and we are not going to propagate a half-remembered
number into a design document. If it matters for the writeup, look it up.

Whatever is chosen here is chosen at 400M. The 600M transfer runs in Tier 5
have a different critical batch, and the writeup has to say the batch was picked
at the small size.

---

## Tier order note

Order is Tier 0b (batch), then Tier 1 (optimizer), then Tier 0 (noise floor),
then the rest. The noise floor has to be measured
on the optimizer and LR everything else will use, otherwise sigma is measured
against one base and the arms are compared against another.

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

## Tier 1 - settle the optimizer (two waves, 24 runs)

Runs first. Everything downstream is a single-factor arm off the winner.

Only AdamW and NorMuon. Muon, NorDion2 and stable_adamw are dropped.

**NorMuon sits at dion's defaults, not nanoplm's.** nanoplm deviates from the
library on four knobs, none of them documented as deliberate:

| knob | dion | nanoplm | base now |
|---|---|---|---|
| `adjust_lr` | spectral_norm | rms_norm | spectral_norm |
| `cautious_wd` | false | true | false |
| `nesterov` | false | true | false |
| `epsilon` | 1e-8 | 1e-7 | 1e-8 |
| `weight_decay` | 0.01 | 0.01 | 0.01 |
| `lr` | 0.01 | 1e-3 | swept |

Verified by building the real optimizer from the config and diffing every
param-group value against `NorMuon.__init__` defaults: zero deviations.

AdamW stays at stock ModernBERT (beta2 0.98, wd 1e-5), since that is the
paper's baseline. Its wd and beta2 become arms in Wave 2 rather than
assumptions.

### Wave 1 - learning rate (10 runs)

| arm | grid | runs |
|---|---|---|
| AdamW, stock ModernBERT | 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4 | 5 |
| NorMuon, dion defaults | 2.5e-3, 5e-3, 1e-2, 2e-2, 4e-2 | 5 |

The NorMuon grid is centred on dion's own default of 1e-2, which is the LR that
default was chosen for under `spectral_norm`.

**If the winner sits at either end of a grid, the optimum is outside it and the
grid gets extended before anything is built on it.**

### Wave 2 - weight decay and beta2, at Wave 1's winning LR (14 runs)

Depends on Wave 1. Do not submit blind.

| optimizer | knob | values | base | runs |
|---|---|---|---|---|
| AdamW | `adam_weight_decay` | 0.01, 0.1 | 1e-5 | 2 |
| AdamW | `adam_beta2` | 0.95, 0.999 | 0.98 | 2 |
| NorMuon | `muon_weight_decay` | 1e-5, 0.1 | 0.01 | 2 |
| NorMuon | `muon_beta2` | 0.9, 0.98 | 0.95 | 2 |
| NorMuon | `muon_cautious_weight_decay` | true | false | 1 |
| NorMuon | `muon_nesterov` | true | false | 1 |
| both | winners re-run for confirmation | | | 4 |

The last two single runs are worth their cost: nanoplm ships cautious decay and
nesterov ON while dion ships them OFF, so somebody thought they mattered. Either
they win, which vindicates the nanoplm default, or they do not and the base
stays at the library default with evidence behind it.

---

## Tier 2 - one factor at a time (about 65 runs)

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
| A3 | srelu | `mlp_activation` | squared ReLU, ungated. Cheap, and the only other value the config accepts |
| A4 | QK norm on | `use_qk_norm` | attention-logit stability at depth |
| A5 | QK norm before RoPE | `reorder_RoPE_QKNorm` | order is not obviously settled |
| A6 | all-global attention | `attn_layer_pattern` | is alternating local/global earning its place |
| A7 | rope theta 10k | `global_rope_theta` | 160k is inherited, not tuned for 512-token proteins |
| A8 | GQA, 8 kv heads | `num_kv_heads` | cheaper attention, more tokens in 6 h |
| A9 | MLM 15% | `mlm_probability` | 30% is high |
| A10 | MLM 40% | `mlm_probability` | the other direction |
| A11 | mask 100%, no 80/10/10 | `mask_replace_prob` | the 10/10 split is cargo-culted from BERT |

Struck: tied embeddings (the base is untied and we are not testing it back) and
span masking. `eval_mlm_masking_strategy: token` stays pinned, and token
masking is now the only strategy anyone runs.

Weight decay and beta2 moved to Tier 1, where they belong: they are optimizer
knobs and both optimizers need their own.

`mlp_activation` accepts only `{swiglu, geglu, srelu}`, and `srelu` is
`relu(x).square()` (`config.py:513`). So A2 and A3 are the complete set of
alternatives to the geglu base, not a sample of one.

### 2b. MoE and canon layers

Runs after 2a and before 2c, because we expect to ship both and the question is
how to configure them, not whether to use them. These are small grids, not
single arms.

**MoE. Matched ACTIVE parameters** (decided), **swiglu** (decided, and also
forced: `use_moe=true` refuses geglu, `config.py:432`).

**There is always a shared expert.** `MoELayer` builds one
`ModernBertSwiGLUMLP(config)` that processes every token (`moe.py:468`), at the
same `intermediate_size` as a routed expert, with no knob to disable it. So:

- active MLP width = `(moe_top_k + 1) x intermediate_size`
- sparsity = `(moe_num_experts + 1) / (moe_top_k + 1)`

That is why jul30's 48 routed experts at top_k 3 were reported as 49 total and
12.25x, not 16x. Any grid that ignores the shared expert is wrong on both axes.

Matched active means `(top_k + 1) x expert_inter = 2688`.

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
-0.0080 from 8.25x to 12.25x while total parameters grew 44% (0.95B to 1.37B).
And jul30 never measured a noise floor, so -0.0080 may well be noise.

Keeping 8x costs 2 runs and re-tests that conclusion against a sigma we will
actually have. If 12x wins again by more than 2 sigma, it is settled properly.
If it does not, we ship the model that is 1B parameters smaller for free.

The control is dense swiglu (arm A2 from 2a), not the geglu base.

Two known issues to handle: quack GEMM rejects bf16 so eval falls back to
cutlass, and the CUTLASS grouped-GEMM JIT has a multi-rank build race, so
prebuild the `.so` before `srun`.

**Canon layers.** `canon_layers_mode` x `canon_layer_set`.

By running after 2a we know the norm result. The CuTe canon backend covers
rms_conv and bare conv only, so on a layernorm base canon takes a slow Triton
fallback and would lose on wall-clock for a kernel reason rather than an
architectural one. If rmsnorm (A1) wins in 2a, this resolves itself. If it does
not, canon runs handicapped and the writeup has to say so.

### 2c. Our own ideas

These need a second seed before anyone should believe them. That is +10 runs at
each arm's best LR, counted in the budget.

| id | change | key |
|---|---|---|
| C1 | residual lambdas | `use_resid_lambdas` |
| C2 | x0 lambdas | `use_x0_lambdas` |
| C3 | ProRes | `use_prores` |
| C4 | paired-head attention | `use_paired_head_attention` |
| C5a | mHC-lite, layer boundary | `use_mhc_lite` + `mhc_lite_wrapping_level: layer` |
| C5b | mHC-lite, sublayers | `use_mhc_lite` + `mhc_lite_wrapping_level: sublayers` |
| C6 | NOBLE | `use_noble` |
| C7 | Loopie | `use_loopie` |
| C8 | Huginn recycling | `use_huginn_looping` |

Struck: RePO.

P3 is resolved and C1/C2 are unblocked. The lambdas are fp32 parameters that
FSDP casts to bf16 for the forward, with fp32 masters in the optimizer. That is
the ordinary mixed-precision path every other weight takes, not the
`cast_forward_inputs` problem that broke the RoPE angles. Values near 1.0 have
~0.4% spacing in bf16 and updates accumulate in the fp32 master. Verified by
building the model under a real 1-rank FSDP2 with `MixedPrecisionPolicy` and
printing the dtypes.

**C5a vs C5b (mHC-lite placement).** DSv4 and GLM 5.3 Flash apply it at the
*sublayer* level, attention and MLP separately; nanoplm defaults to once per
*layer* boundary. Both are tested rather than assumed.

No code change needed:
`mhc_lite_wrapping_level: Literal["layer", "sublayers"] = "layer"` already
exists (`config.py:127`), and the config refuses a non-default level unless
`use_mhc_lite` is on, so the two arms cannot be mis-specified silently.

MoE and canon layers moved to their own tier, 2b, which runs before this one.

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
| 0b | 6 | global batch size, runs first. SUBMITTED |
| 1 | 24 | optimizer: Wave 1 LR (10, blocked on 0b), Wave 2 wd/beta2 (14, blocked on Wave 1) |
| 0 | 6 | noise floor on the winning optimizer + LR |
| 2a | 33 | 11 arms x 3 LRs |

| 2b | ~8 | MoE sparsity x granularity, canon mode x set |
| 2c | 27 | 9 arms x 3 LRs |
| 2c seeds | 9 | second seed on our own ideas |
| 3 | ~16 | greedy ladder |
| 4 | ~12 | leave-one-out |
| 5 | 6 | transfer + fp8 |
| **total** | **~136** | **~12,600 GPU-hours** |

Strike rows if that is too many. The tiers are ordered so cutting from the
bottom of 2c costs the least.

## Considered, not included

Say the word and any of these come in. Each is 2 to 5 runs.

- sequence length 1024 instead of 512
- warmup length (currently 1000 steps)
- decay length and shape (currently 30 min, 1-sqrt). We flagged 30 min as short
  versus the usual 10 to 20%, but did not test it.
