# Plan

What is left to run. Done work is in [results.md](results.md); the matching and
threshold rules are in [method.md](method.md).

Every architecture arm: 6 h stable phase, 4 nodes, 16 GH200, 4.19M tokens/step
(grad_accum 4), ~10,568 steps, ~44.3B tokens, about 96 GPU-hours. Compute-neutral
arms run fixed-step instead. Each arm changes exactly one thing from the base and
runs at **3 LRs** (0.5x / 1x / 2x its expected optimum), 1 seed. The LR curve is
the robustness check: an arm that only wins at one LR did not win.

## Now: Tier 1d, the LR re-check at 20% masking (`t1d-`)

4 runs, fixed-step at `max_steps: 5000`, LRs 3.5e-3 / 5e-3 / 7e-3 / 1e-2.
Submitted 2026-09-10 as jobs 1750791, 1750793, 1750794, 1750795. Masking changed
the task, so 7e-3 (tuned at 30%) has to be re-confirmed at 20% before the base
moves. First use of the fixed-step rule.

## Tier 2a: standard recipe knobs (24 runs, 8 arms x 3 LRs)

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

`mlp_activation` accepts only `{swiglu, geglu, srelu}`, and `srelu` is
`relu(x).square()` (`config.py:513`), so A2 and A3 are the complete set of
alternatives to the geglu base rather than a sample.

Struck: tied embeddings (the base is untied and we are not testing it back) and
span masking, since `eval_mlm_masking_strategy: token` is pinned and token
masking is now the only strategy anyone runs. Weight decay and beta2 moved to
the optimizer tier, where they belong.

## Tier 2b: MoE and canon layers (~8 runs)

After 2a and before 2c, because we expect to ship both and the question is how
to configure them, not whether to use them.

**MoE**, matched active parameters and swiglu (also forced: `use_moe=true`
refuses geglu, `config.py:432`). **There is always a shared expert**:
`MoELayer` builds one `ModernBertSwiGLUMLP(config)` that processes every token
(`moe.py:468`) at the same `intermediate_size` as a routed expert, with no knob
to disable it. So active MLP width is `(moe_top_k + 1) x intermediate_size` and
sparsity is `(moe_num_experts + 1) / (moe_top_k + 1)`. That is why jul30's 48
routed experts at top_k 3 were 49 total and 12.25x, not 16x. Any grid that
ignores the shared expert is wrong on both axes.

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

**Canon layers**: `canon_layers_mode` x `canon_layer_set`. Running after 2a
means we know the norm result, which matters because the CuTe canon backend
covers rms_conv and bare conv only. On a layernorm base canon falls back to slow
Triton and would lose on wall clock for a kernel reason rather than an
architectural one. If rmsnorm (A1) wins in 2a this resolves itself; if not,
canon runs handicapped and the writeup says so.

## Tier 2c: our own ideas (27 runs + 9 second seeds)

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

Struck: RePO. These need a second seed before anyone should believe them, which
is the +9 runs at each arm's best LR.

C1/C2 are unblocked: the lambdas are fp32 parameters cast to bf16 for the
forward with fp32 masters in the optimizer, the ordinary mixed-precision path,
not the `cast_forward_inputs` problem that broke the RoPE angles. Values near
1.0 have ~0.4% spacing in bf16 and updates accumulate in the fp32 master.
Checked under a real 1-rank FSDP2 with `MixedPrecisionPolicy`.

## Tier 3: greedy ladder (~16 runs)

Take the Tier 2 winners and add them one at a time in descending effect size,
2 seeds per rung. A rung stays only if it beats the previous rung by more than
2 x sigma_seed. This is where interactions show up: norm, activation and
optimizer are the three most likely to interact, and one-at-a-time cannot see
that.

## Tier 4: leave-one-out (~12 runs)

For each ingredient in the final recipe, run the recipe without it, 2 seeds. An
ingredient that does not hurt when removed does not go in the paper's recipe,
however well it did alone. This is what catches ingredients that were only ever
measuring each other.

## Tier 5: transfer and precision (6 runs)

4 runs of base vs final recipe at 574.8M (h1152/L36), 2 seeds each, plus 2 runs
of fp8 vs bf16 on the final recipe. A win that does not survive the scale-up
does not go in the paper.

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

| tier | runs | note |
|---|---|---|
| 1d | 4 | LR re-check at 20% masking. RUNNING |
| 2a | 24 | 8 arms x 3 LRs |
| 2b | ~8 | MoE sparsity x granularity, canon mode x set |
| 2c | 27 | 9 arms x 3 LRs |
| 2c seeds | 9 | second seed on our own ideas |
| 3 | ~16 | greedy ladder |
| 4 | ~12 | leave-one-out |
| 5 | 6 | transfer + fp8 |
| **left** | **~106** | of ~140 for the whole series, ~12,700 GPU-hours |

Strike rows if that is too many. The tiers are ordered so cutting from the
bottom of 2c costs the least.

## Still open

- **Downstream numbers are development mode.** The reference ESM C report is
  too, so the comparison is like-for-like, but neither is reportable. A
  `development_mode: false` run is owed.
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
