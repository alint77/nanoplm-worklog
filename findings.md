# Findings

Things that cost a run, or would have.

## Reading a number

### Node-speed variation is ~9% and it outweighs the effects we are measuring
Ten Wave 1 runs, identical except learning rate, landed on different nodes and
ran at median step times from 1826 ms to 1998 ms. A 9.4% spread with no
architectural cause.

Under wall-clock matching that becomes a token-budget difference: 9880 steps on
the fastest allocation versus 9020 on the slowest, 860 steps apart. At the
late-run slope of -0.0085 loss per 1000 steps, those 860 steps are worth
**0.0073 loss**.

That is larger than the effects we are trying to resolve. Adjacent LR points in
Wave 1 differ by 0.002 to 0.005, so node placement alone can reorder them, and
it did: at 6 h wall clock NorMuon 5e-3 looked best (2.2142, on the fastest
nodes), but at equal steps 7e-3 wins (2.2163 vs 2.2184).

**Rule for the series:**

- Arms whose compute per step is IDENTICAL (learning rate, weight decay, beta2,
  seed, masking rate, rope theta, norm type) must be compared **at equal
  steps**, not at the wall-clock stop. Their throughput differences are pure
  node noise.
- Arms that genuinely change compute per step (GQA, MoE, activation, attention
  pattern, sequence length) are compared at the wall-clock stop, because there
  the throughput difference IS the arm and the design intends it to count. But
  node noise of ~0.007 still sits on top and has to be part of sigma.
- Every run logs eval vs step, so both comparisons are available after the
  fact. Report both.

This also means sigma_seed, when Tier 0 measures it, will contain a node-speed
component unless those runs are compared at equal steps too. Measure it both
ways.

### Wall-clock matching contaminates downstream scores through step count
Found 2026-09-10, by the user reading the Tier 1b table: the optimizer knobs
have no compute overhead, yet their checkpoints sit at different steps
(10303-10975, a 672-step spread) purely because of node-to-node speed. Eval
loss is protected from this by the common-step column. **Downstream scores are
not**: they come from whatever checkpoint the arm happened to end on.

Two independent estimates of the size of the effect agree:

- **Predicted**, from the cross-arm slope d(long P@L)/d(loss) = -1.972 times
  the per-step loss decline -7.0e-6: **1.38e-5 per step**.
- **Measured**, regressing long P@L on checkpoint step across the six base
  replicates: **1.26e-5 per step, Pearson r = +0.890**.

r = 0.89 among *identical configurations*. The only thing that differs between
those six runs is how many steps their node allowed.

Correcting the replicates with the independent coefficient (so the test is out
of sample, not fitted on the points it corrects) cuts their spread:

| | 2 sigma | spread |
|---|---|---|
| raw | 0.0048 | 0.0068 |
| corrected to step 10500 | **0.0022** | 0.0024 |

**Step spread accounted for 54% of what was being reported as the downstream
noise floor.** The real floor is roughly half what `method.md` states.

Corrected, three Tier 1b verdicts flip from tie to separated. Cautious weight
decay is the starkest: it ran 733 fewer steps than the base, and goes from
-0.0034 to +0.0058 once that is accounted for.

**That is not a result.** It says the raw comparison is confounded, not that
cautious wins. The correction is shaky too: its threshold rests on six points,
and cautious stopped at 10303, which is 321 steps below the lowest replicate
(10624), so it gets the biggest correction on the least support.

**The loss verdicts are unaffected.** Tier 1b's decisions were made on the
common-step loss column and stand as recorded.

**Does a fixed step put arms in different schedule phases?** Not here. Every
arm in t0/t1a/t1b/t1c runs `warmup_stable` and is still on the plateau at
10500: `muon_lr` reads exactly 7.00e-03 at both step 10300 and 10500 in every
arm checked. Step 10500 is therefore a clean comparison point. It would not be
for the decay campaign, where the comparable milestone is end-of-decay, not a
step number.

**Rule, from here on:** every arm writes a checkpoint at the pre-registered
common step in addition to its wall-clock one, so downstream scores are at
equal steps by construction. Cost is 3.0 GB per run as the pipeline currently
saves (1.6 GB weights + 1.6 GB optimizer); only the weights are needed for
eval, and there is no model-only save option today, so either add one or prune
the optimizer state afterwards.

### Classify an arm as compute-changing from the TRACE, never from intuition
I wrote that cautious weight decay cost ~5% throughput and therefore lost on
wall clock. That was wrong, and the way it was wrong is instructive.

Cautious decay ran at 2091 ms/step against ~1980 ms for its siblings, and it was
the only Wave 2 run on jpbo-013. The trace shows the arm is not responsible:

- GEMM: 11,861 launches in both runs (identical work), 680.5 vs 676.6 us per
  launch, ratio 1.006. The GPUs are the same speed.
- `multi_tensor_apply`, where the optimizer's elementwise work lives: 75.2 vs
  75.7 ms. Identical. Cautious decay adds nothing measurable.
- compute union: 13,286 vs 13,436 ms. Cautious does 1% LESS compute.
- exposed NCCL: 174.2 vs 89.8 ms per step.

The 110 ms step difference is the 84 ms/step of extra exposed comms. It is a
network or straggler problem on that node group.

**Two rules follow.**

1. Whether an arm is "compute-changing" is a measurement, not a guess. Compare
   compute union in the trace against the control. Only then is the wall-clock
   comparison the right one. Applying the wall-clock rule on the assumption that
   an arm is slower will attribute node noise to the arm, which is exactly what
   happened here.
2. **Exposed comms varies 2x between node groups** (89.8 to 174.2 ms/step, 4.5%
   to 8.3% of the step). That is a per-run lottery unrelated to the arm, and it
   explains the 9.4% step-time spread seen across identical configs in Wave 1.
   Default to comparing at equal steps; reserve wall-clock for arms whose
   compute union actually differs.

Practical follow-up: log exposed comms per run and flag any run whose compute
union matches the control but whose step time does not. That is a bad node, and
it should be resubmitted rather than interpreted.

### Gate every run on its trace at ~5 minutes, not at 6.5 hours
The profiler window is steps 15-23, so the trace lands about 5 minutes in
(compile dominates). A run that drew a bad node group can therefore be detected
and requeued for ~7 minutes of waste instead of 6.5 hours.

**The metric has to be self-normalizing**, or an arm that legitimately does more
compute looks degraded. Step time cannot do it. This works:

    exposed_comms / compute_union

An arm doing more compute grows the denominator AND hides comms better, so a
high ratio always means comms-bound, never "this arm is heavy".

Measured over 24 Tier 1 traces:

| verdict | count | ratio |
|---|---|---|
| healthy | 21 | 1.4% to 6.6% |
| degraded | 3 | 9.1% to 10.5% |

The gap between 6.6% and 9.1% is empty, so the threshold sits at 8%.

**Bad nodes repeat.** `jpbo-001-[36-37,41,43]` flagged twice, on two different
arms. `jpbo-013-[05,08,11,14]` once. So an exclude-list is worth keeping rather
than just retrying blind: `run/bad_nodes.txt`, seeded with those 8 nodes.

Note a different subset of the same rack, `jpbo-001-[09,11-12,15]`, came in at
5.3%. The problem is specific nodes, not the rack.

**Tooling:** `tools/trace_gate.py` (analyse one trace, exit 1 if degraded) and
`tools/submit_gated.sh` (submit, wait for trace, requeue elsewhere on failure,
up to MAX_TRIES, accumulating the exclude list).

**This does not retroactively invalidate the Tier 1 results.** Equal-step
comparison already removes the throughput penalty, so a slow node costs tokens,
not loss-per-token, and the LR and weight-decay rankings were read at equal
steps. Only the wall-clock column was affected, which is why the cautious
verdict was wrong.

### dt in the log is a windowed average and includes eval
Step-time distribution over a 920-step run at `eval_steps: 50`:
`min 506, p25 512, median 514, p75 706, max 1796`.

The median is the real step time. The 706 values are logging windows that
contain an eval, and 1796 is the first window, which contains compile. Reading
the last line of a log as "the step time" will be wrong whenever that window
happened to include an eval.

### Eval is not comparable across MLM-rate arms, and it is noisy
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

### Eval costs 4.0 s a time
Measured off a real 4-node log, not estimated. At `eval_steps: 250` over a 6 h
run that is about 169 evals, roughly 11 minutes, 3% of the budget. Now at 500.

### A run has no final eval unless the step count lines up
Eval fires on `global_step % eval_steps == 0 or at_wsd_stable_end`, and there
is no eval after the training loop. So a decay run whose `decay_steps` is not a
multiple of `eval_steps` finishes with no score at all. Round it.

### Bigger global batch buys scale-out headroom, not comms savings
FSDP2 does skip the reduce-scatter during gradient accumulation
(`set_requires_gradient_sync(at_accum_boundary)` in `pure_pipeline.py`), so a
bigger batch really does cut collectives per token. It just does not matter
here.

Measured on the h1024/L32 4-node trace, subtracting the compute-kernel union
from the NCCL union:

- NCCL total (union): 140.4 ms per step
- NCCL **exposed**: 5.2 ms per step, 1.0% of a 511 ms step

Upper bound saving is `(ga-1)/ga x 5.2 ms`: 2.6 ms at ga=2, 3.9 ms at ga=4.
Under 1% of throughput either way.

The lesson repeats an earlier one: total NCCL time (140 ms) looks like a big
lever and is not. Only exposed comms is a cost.

**The real reason to want a bigger global batch is strong scaling.** Global
batch caps the world size that can run at full local batch: at
`micro_batch_seqs` 128 and 512-token sequences each GPU takes 65,536 tokens, so
1M tokens caps us at 16 GPUs, 2M at 32, 4M at 64, 8M at 128. Past that ceiling
the local batch shrinks, GEMMs get smaller and utilisation drops. That is an
argument about the eventual full-scale run, and it is arithmetic rather than
something to measure.

## Precision, kernels and profiling

### FSDP silently turns fp32 layer inputs into bf16
`MixedPrecisionPolicy.cast_forward_inputs` defaults to `True`. Every
floating-point tensor passed to a wrapped module's forward is cast to
`param_dtype`. With per-layer sharding that is every encoder layer.

So any "this is fp32" assumption at a layer boundary is wrong today. Values in
[-1, 1] like cos/sin do not care. Raw angles do: at position 8191 the angle is
about 8191 radians, and bf16's 0.4% relative error is tens of radians.

Fix: hold such tensors as buffers. Buffers are not forward inputs.

Still to check: the residual-lambda tensors are passed as forward args, so they
are bf16 right now. Resid lambdas is an arm, so confirm before it runs.

### The lambdas are bf16 in the forward, and that is fine
`resid_lambdas` and `x0_lambdas` are fp32 parameters. Under FSDP2 with
`param_dtype=bf16` they are bf16 in the forward with fp32 masters in the
optimizer, which is the ordinary mixed-precision path every weight takes. It is
NOT the `cast_forward_inputs` problem that broke the RoPE angles, because those
were non-parameter tensors passed as forward args at large magnitudes.

Values near 1.0 have ~0.4% spacing in bf16 and updates accumulate in the fp32
master. Verified under a real 1-rank FSDP2 with `MixedPrecisionPolicy`.

### TE fused RoPE is faster and buys nothing
4 nodes, h960/L30:

| | eager | TE fused |
|---|---|---|
| rope kernel time (8 steps) | 486.4 ms | 260.9 ms |
| NCCL time | 627.8 ms | 1704.3 ms |
| step time | 470.5 ms | 470.6 ms |

Rope was already overlapping communication, so it was never on the critical
path. Removing work that is not the bottleneck moves nothing. It was neutral at
h960, the width where the eager path is worst, so it cannot help at h1024.

### TE's RoPE cost scales with the table length, not the sequence length
Same call, 128x512 packed, 15 heads:

| table rows | time |
|---|---|
| 512 | 0.287 ms |
| 2048 | 0.388 ms |
| 8192 | 1.274 ms |

Sizing the table at `max_position_embeddings` while running 512-token sequences
cost 4.4x and made the step 49% slower.

### Why NCCL time ballooned in the TE RoPE run
It did not. NCCL kernel *duration* includes the time a rank sits inside the
collective waiting for its peers. The fused kernel made the compute before the
all-gather faster, so ranks arrived earlier and waited longer, and the recorded
NCCL time grew from 628 ms to 1704 ms while step time did not move at all.

The collective was the critical path the whole time. Total NCCL time is not a
cost; only exposed (non-overlapped) comms is.

### A silent fallback is invisible in a profile
The compiled eager rope is named
`triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_view_*`. So "no TE
kernels in the trace" and "TE kernels present" look the same by absence.

Our first A/B reported a clean null result while the fused path had never run
once. Anything with a fallback has to say out loud which path it took.

### Two host syncs per micro-step
`num_valid_tokens` was a Python int. `_move_batch_to_device` only moves
tensors, so it survived to the forward and became a CUDA tensor there: a
blocking copy from pageable memory, one `cudaStreamSynchronize` per micro-step.
The non-finite check did an `all_reduce().item()` per micro-step as well.

Fixed both. 1 node -4.6%, 4 nodes -1.3%, host-blocked time to zero.

### Width sets MFU, not depth
Changing depth 21% (L28 to L34 at h896) moved MFU 0.2 points. Changing width
moved it 5. Comms share was flat and exposed comms was actually lower at the
larger size.

Also: only non-overlapped comms is a cost. Adding up total comms time is wrong.

## Optimizer

### NorMuon's LR scaling is shape-blind under rms_norm
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

### nanoplm deviates from dion's optimizer defaults on four knobs
`adjust_lr` (rms_norm vs spectral_norm), `cautious_wd` (true vs false),
`nesterov` (true vs false), `epsilon` (1e-7 vs 1e-8). None is documented as
deliberate. The series now runs dion's defaults, and the last two are arms in
Tier 1 Wave 2 so the deviation gets tested rather than inherited.

## MoE kernels

### MoE always has a shared expert, and it changes the arithmetic
`MoELayer` builds one `ModernBertSwiGLUMLP(config)` that processes every token
(`moe.py:468`), at the same `intermediate_size` as a routed expert. There is no
knob to turn it off.

So active experts per token is `top_k + 1`, not `top_k`, and:

- active MLP width = `(top_k + 1) x intermediate_size`
- sparsity = `(moe_num_experts + 1) / (top_k + 1)`

This is why jul30's `arm11-moe12x` (48 routed, top_k 3) was reported as 49
experts and 12.25x. A matched-active grid that counts only routed experts is
wrong on both axes: it under-counts active width by 25% at top_k 3.

### Nothing beats sonicmoe on Hopper, and the reason is not the GEMM
Measured on one GH200, one MoE routed-expert path, fwd+bwd, eager, 65536
tokens/GPU, at the four ladder cells (ms):

| cell | E | top_k | I | cutlass | torch._grouped_mm | TE 2.18 | sonicmoe |
|---|---|---|---|---|---|---|---|
| g4-S8  | 31 | 3 | 672 |  9.655 |  8.984 |  9.059 | **4.546** |
| g4-S12 | 47 | 3 | 672 |  9.939 |  9.123 |  9.200 | **4.720** |
| g8-S8  | 63 | 7 | 336 | 13.322 | 11.898 | 12.179 | **5.930** |
| g8-S12 | 95 | 7 | 336 | 13.421 | 12.058 |  FAIL  | **6.193** |

All four parity-clean first (`_grouped_mm` matches cutlass exactly on out, dWi
and dWo; sonicmoe within 1%, bf16 reduction order).

sonicmoe fuses the dispatch gather into Wi's A-load and the scatter plus
router-weight combine into Wo's epilogue. Every other grouped-GEMM backend
needs `moe_scatter_dispatch` and `moe_gather_combine` around it: at g8-S12
that is a materialized (458752, 1024) bf16 tensor, ~940 MB, touched about six
times across forward and backward. That traffic is the gap, and swapping the
GEMM cannot close it.

The published record agrees: SonicMoE (ICLR 2026) beats ScatterMoE by 1.86x,
MoMoE, MegaBlocks, Megatron and DeepGEMM++ on H100, at intermediate size 256,
which is our 336 regime.

`torch._grouped_mm` is still worth knowing about. It is in our torch 2.12, runs
on GH200 sm90 bf16 with device-side offsets, handles ragged and empty groups,
and profiles as one CUTLASS sm9x grouped kernel per call with zero Memcpy DtoH.
It beats our JIT-built cutlass extension by 7 to 11% with no dependency and no
multi-rank build race.

### Expert width does not matter, top_k does
One GH200, one MoE routed-expert path, fwd+bwd, eager, bf16, sonicmoe backend,
65536 tokens (same setup as the table above). Active MLP width is
`(top_k + 1) x intermediate_size`; dense is 2688, so every row except 512 is an
exact active-parameter match.

| active | top_k | inter | E (S8) | ms S8 | E (S12) | ms S12 | 128-aligned |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 896 | 23 | 12.39 | 35 | 12.64 | yes |
| 4 | 3 | 672 | 31 | **12.43** | 47 | **12.58** | no |
| 5 | 4 | 512 | 39 | 12.05 | 59 | 12.17 | yes |
| 7 | 6 | 384 | 55 | 12.63 | 83 | 13.10 | yes |
| 8 | 7 | 336 | 63 | 13.41 | 95 | 13.96 | no |

(An eager dense SwiGLU at the same shape is 12.62 ms, for scale. Training
compiles the model, so that is not the dense arm's real step cost and no
"MoE beats dense" claim rests on it.)

Every number above is the min of three 10-iteration timings in a **fresh
process per cell**, after the single-process version of this sweep was caught
producing garbage (below). The two runs agree within 1.4% on all ten cells.

Two results.

**128-byte alignment of the expert width predicts nothing.** 672 is not a
multiple of 128 and ties the aligned 896, and beats the aligned 384. Every
width builds and runs on sonicmoe, 336 and 672 included, so there is no
legality constraint either. (The cutlass probe in the same script died on
missing CUTLASS headers while the sonicmoe probe ran, which incidentally proves
sonicmoe was not silently falling back.)

**`top_k` predicts it, monotonically, at both sparsities.** Going 2 -> 7 active
routed experts costs 8.2% at S8 and 10.4% at S12 at constant active FLOPs. That
is dispatch and combine traffic scaling with token replication, the same
mechanism that makes sonicmoe win in the first place: the fused gather/scatter
is cheaper than materializing the dispatch buffer, but it is not free, and it
grows with `top_k`.

65536 tokens is exactly the training micro-step: `max_seq_len` is 512 and
`micro_batch_seqs` is 128, so each GPU sees 128 x 512 = 65536 tokens per
micro-step and `global_batch_tokens: 4194304` over 16 GPUs comes out as
`grad_accum = 4`. The table above is therefore at the shape that actually runs,
not a proxy for it.

The penalty is strongly token-count dependent, which is why that matters: 336
is only +1.8% over 672 at 8192 tokens and +8.5% at 65536. At 262144 (a full
optimizer step's worth, which no single forward ever sees) the ordering holds:
dense 49.15, g5-S8 45.58, g4-S12 47.08, g7-S12 49.34, g8-S8 51.02, g8-S12 53.83.

### Benchmark one cell per process, or the tail of the sweep is fiction
The 262144 sweep was first run as one process looping over cells with
`del m; torch.cuda.empty_cache()` between them. Its first six rows were right
and its last three were not: g8-S12 came out at 92.04 ms and g5-S8 at 77.48,
against 53.83 and 45.58 when each is run alone. That is +71% and +70% of pure
fiction, non-monotone against every neighbouring cell, from allocator
fragmentation accumulating across cells with large working sets. `empty_cache()`
does not undo it.

It was caught only because the numbers broke monotonicity in `top_k`, which the
rest of the data had already established. Without that prior there was nothing
in the output to distinguish the bad rows from the good ones, and two of them
would have gone into a decision. The 65536 table above was re-run the same way
as a check and came back clean, so nothing built on it moved, but the lesson
stands: **one cell per process for anything that decides something**, and build
an ordering prior you can sanity-check the tail against.

`inter 512` at top_k 4 is dominated. It is the only row that is not an exact
active match (2560, -4.76%), and per active FLOP it is 1.8% slower than 672 at
S8 and 1.6% slower at S12. It buys nothing.

One knock-on: sonicmoe's fused top-k kernel is gated on `E <= 4096 and K <= 16
and E % 8 == 0` (`sonicmoe/functional/forward.py`). Every cell in the grid has
odd E (shared expert makes it `S*(top_k+1) - 1`), so all of them take the torch
`topk` fallback. It is a router-sized op on (T, E) and did not show up as a
difference between cells, so this is a note, not a problem.

A note on the total-parameter column, since I briefly got this wrong: counting
with `moe_leading_dense_layers: 0` gives 2.25B/3.31B against plan.md's
2.13B/3.12B, which looks like a 5% error in the plan. It is not. The plan
assumed two leading dense layers, and at the adopted
`moe_leading_dense_layers: 2` the instantiated model counts 2.135B and 3.126B,
matching it. The two dense layers are built at
`moe_dense_intermediate_size = (top_k + 1) * intermediate_size`
(`modeling.py:959`), which is 2688 in both cells, so they are exactly base
MLPs and active width stays matched across every layer.

### MoE OOMs at micro_batch_seqs 128, and an eager per-layer probe could not see it
The worry was that MoE would OOM at `micro_batch_seqs: 128` and force 64. It
does not, and the reason is the same fusion that makes sonicmoe fast. Activation
bytes kept for backward, one MLP or MoE layer, 65536 tokens, bf16, measured on
one GH200:

| layer | saved for bwd | x32 layers | routed weights bf16, x32 |
|---|---|---|---|
| dense inter 2688 | 1.47 GB | 47.00 GB | 0.53 GB |
| g4-S8  (k=3, 672) | 0.97 GB | 30.93 GB | 3.94 GB |
| g4-S12 (k=3, 672) | 0.98 GB | 31.21 GB | 5.91 GB |
| g7-S8  (k=6, 384) | 0.91 GB | 29.25 GB | 3.94 GB |
| g7-S12 (k=6, 384) | 0.93 GB | 29.69 GB | 5.91 GB |
| g8-S8  (k=7, 336) | 0.91 GB | 29.07 GB | 3.94 GB |
| g8-S12 (k=7, 336) | 0.92 GB | 29.54 GB | 5.91 GB |

A dense SwiGLU MLP saves its (T, 2 x 2688) preactivation. sonicmoe fuses
up/act/down and never materializes the equivalent, so it saves about a third
less despite the same active width. Across 32 layers that is 16-18 GB *back*.

Against that, MoE adds unsharded weight bytes. `fsdp_reshard_after_forward` is
`false`, so every layer's parameters stay materialized between forward and
backward: +5.4 GB at S12 over dense. Optimizer state is sharded, and at 3.31B
over 16 GPUs is 207M params/GPU, roughly +2.2 GB over dense under NorMuon's
three fp32 buffers. Call it +7.6 GB of state against -17 GB of activations.

From that I predicted MoE would sit *below* the dense baseline in peak memory
and that 128 would hold. **It does not. g7-S12 OOMs at 128**, on job 1772818,
with 90.89 GiB allocated on a 95 GiB card, failing on a 768 MiB allocation
during the first compiled forward.

The prediction was wrong because the probe was eager and the training loop
compiles. Inductor materializes, per MoE layer, exactly the buffers sonicmoe's
fusion avoids in eager:

    buf52 = (T*top_k, inter)    = (393216, 384)   288 MB
    buf53 = (T*top_k, 2*inter)  = (393216, 768)   576 MB
    buf57 = (T*top_k, hidden)   = (393216, 1024)  768 MB

at `T*top_k = 65536 * 6`. The fusion is real and the compiled graph does call
the fused kernels (`quack.gemm_gated_out`, `quack.gemm_out`,
`sonicmoe._router_forward` are all in the generated code, so there is no silent
fallback to cutlass), but inductor still allocates the operands and keeps the
ones the grouped-GEMM backward needs. An eager single-layer probe cannot see
any of this, and neither can a 32x extrapolation from it.

The general form of the error: **a per-layer eager measurement does not predict
a compiled model's peak memory**, and "saved for backward in eager" is not the
quantity that decides whether a run fits. The only instrument that answers the
memory question is the run itself.

`top_k` is the multiplier on all three buffers, which is the second reason to
prefer 4 active experts over 7: at top_k 3 they are half the size.

The fix is `micro_batch_seqs: 64` (grad_accum 4 -> 8) applied uniformly to every
MoE cell, which halves T and therefore all three buffers. Not activation
checkpointing, which would change compute per step.

One caveat on the numbers above: transient peak within a MoE forward is higher
than the saved figure (1.7-2.5 GB vs ~0.95), but it is one layer at a time, so
it adds once to the 32-layer total rather than scaling with it. The bs64 peak
column in that run was allocator-contaminated by the bs128 pass that preceded
it and should be ignored; the `saved` column halved cleanly and is the one the
extrapolation uses.

### The trace gate's threshold does not transfer to MoE
`tools/trace_gate.py` cancelled a healthy MoE smoke (job 1772896) at
`exposed/compute = 15.6%` against its 8% threshold, blacklisted four innocent
nodes, and gave up. The same config re-run ungated on a different draw reached
steady state at 38.4% MFU.

The gate's docstring states its assumption outright: "An arm that genuinely does
more compute grows the denominator and hides comms better, so a HIGH ratio
always means comms-bound, never 'this arm is heavy'." That holds for every arm
tried so far because they all had ~400M parameters, so FSDP all-gather volume
was constant and only the numerator could move. MoE breaks it: 3.13B parameters
is about 8x the weight traffic, so the arm moves the numerator *and* the
denominator, and the numerator moves harder.

Per profiled step (9-step window), MoE g7-S12 against the healthy dense
`t2a-glob` run:

| | dense t2a-glob | MoE g7-S12 | |
|---|---|---|---|
| compute | 13464 ms | 16809 ms | +24.8% |
| nccl | 4559 ms | 6370 ms | +39.7% |
| exposed | 617 ms | 2629 ms | +326% |
| ratio | 4.6% | 15.6% | gate fires |

So 15.6% is this architecture's healthy value, not a bad draw. The dense gate
keeps a 1.74x margin over its healthy 4.6%; transferring the same margin puts
the MoE threshold near 27%. That number is a proposal, not a measurement: it
rests on one healthy MoE draw, and the honest version needs the spread across
two or three draws before it is written into the decision rules. Until then,
**submit MoE arms ungated** rather than adjusting a threshold to fit one run,
which is precisely the post-hoc tuning method.md exists to prevent.

### MoE smoke: the numbers the grid budget needs
g7-S12 (E=83, top_k 6, inter 384, 2 leading dense), `micro_batch_seqs: 64`,
sonicmoe, 4 nodes, job 1773043:

- **2714 ms/step** steady state, against ~2010 ms for the dense base: **+35%**.
  In a 6.5 h arm that is ~8600 steps against dense's ~11600.
- **38.4% MFU.** The jul20 worry that MoE would cost MFU (27% against 36%) does
  not reproduce here.
- **Peak VRAM 70,525 of 72,064 MB, 97.9%.** This is the number to worry about.
  bs64 fits but with almost no headroom, and g7 is the worst cell. Consider
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the MoE arms.
- Router health: dead experts collapse from 89 at step 20 to 1-3 by step 80,
  entropy 0.958, max expert frequency 0.038 against the 1/83 = 0.012 ideal.
  Load balancing is working; no sign of collapse.
- Active parameters per token 402.19M against the dense base's 399.6M. The
  +2.6M is exactly the routers (30 layers x 1024 x 83) and is unavoidable.

### MoE's LR response is flat: the dense 2e-2 transfers
The worry was that 2e-2, tuned on the dense base, had no claim on transferring
to MoE, because the router's gradients and the load-balance loss are new. Four
fixed-step arms on `g4-S12` (E=47, top_k 3, inter 672, bs64), 5000 steps, jobs
1773336-1773339:

| LR | eval loss @5000 |
|---|---|
| 1e-2 | 2.1948 |
| 1.41e-2 | **2.1927** |
| 2e-2 | **2.1927** |
| 2.83e-2 | 2.1936 |

The spread across a 2.83x range of LR is 0.0021, against a 2 sigma eval-loss
floor of 0.0019. The whole sweep is one point. All four arms ran exactly 5000
steps, so no step-spread correction applies.

**Adopted 2e-2**, which ties for best. 1.41e-2 ties it exactly and the standing
"pick the lower one within noise" rule would favour it, but two things outweigh
that here: the dense control runs at 2e-2, so matching it keeps the MoE-vs-dense
comparison free of an LR confound, and loss is a poor instrument for this choice
anyway (across 38 arms it explains 5.6% of the variance in the deciding metric).
Resolving a 0.0000 loss tie on loss would be reading noise.

The winner is not at an edge (1e-2 is worst, 2.83e-2 is mid), so the
extend-the-grid rule does not fire and the bracket is adequate.

**Consequence for the plan: the g7-S12 transfer check is probably not worth two
arms.** A response this flat over 2.83x suggests the optimum is broad. It does
not prove the same for top_k 6, which is a different router regime, so this is a
judgment call rather than a measurement, and it is recorded as one.

Incidental: g4-S12 runs at 2588-2660 ms/step and 39.1-40.2% MFU, slightly better
than g7-S12's 2714 ms and 38.4%. Peak VRAM is 69,580 of 71,336 MB, 97.5%, which
is the same fraction as g7-S12 despite top_k being half. Memory at bs64 is
dominated by the 3.13B of weights and optimizer state, not by the dispatch
buffers, so **both S12 cells are near the edge and the S8 cells should have
room**. Router health held all the way: 0 dead experts, entropy 0.971-0.981.

### Where the MoE step actually goes, and what is wasted
Profiler traces for all four grid arms against the dense `t2a-glob` control.
`compute` is the union of non-NCCL kernel intervals (a GPU measurement, robust
to profiler overhead); `overhead` is steady-state `dt` minus that.

| arm | dt (steady) | MFU | compute | overhead | overhead % |
|---|---|---|---|---|---|
| dense t2a-glob | 2000 ms | 54.0% | 1923 ms | 77 ms | **3.9%** |
| g4-S8 | 2471 ms | 42.0% | 2222 ms | 249 ms | 10.1% |
| g4-S12 | 2639 ms | 39.4% | 2269 ms | 370 ms | **14.0%** |
| g7-S8 | 2579 ms | 40.3% | 2306 ms | 273 ms | 10.6% |
| g7-S12 | 2776 ms | 37.6% | 2475 ms | 301 ms | 10.8% |

So MoE loses on two fronts, and the second one is not about MoE. Compute rises
16-29% over dense despite active parameters being matched, and overhead rises
from 3.9% to 10-14%.

Serial kernel time by category, g4-S12 (per step, 7 profiled steps):

| category | ms/step | % | launches/step |
|---|---|---|---|
| comm: allgather (FSDP params) | 660.2 | 20.8% | 288 |
| gemm: dense (attn proj, head, 2 dense layers) | 644.9 | 20.4% | 4061 |
| moe: expert GEMM down/bwd | 470.7 | 14.9% | 960 |
| norm (triton fused) | 260.9 | 8.2% | 2718 |
| attention (FA3) | 245.2 | 7.7% | 1280 |
| comm: reducescatter (grads, **fp32**) | 217.6 | 6.9% | 36 |
| moe: expert GEMM up+act | 157.1 | 5.0% | 240 |
| elementwise/copy | 140.9 | 4.5% | **9000** |
| moe: expert GEMM bwd (dact) | 119.8 | 3.8% | 240 |
| moe: router combine + routing/sort | 52.1 | 1.6% | 1440 |
| optimizer | 16.8 | 0.5% | 592 |

**The expert GEMMs are not the problem.** They are 748 ms/step at g4-S12 and
950 at g7-S12, and they are the useful work. sonicmoe is doing its job.

**Comm is 28% of all kernel time**, against 25% for dense on a model an eighth
the size. Two specific findings:

*The reduce-scatter runs in fp32.* `fsdp_reduce_dtype: fp32` costs 218 ms/step
of gradient comm at double the bytes bf16 would need. Roughly 110 ms/step, about
4% of step time, is recoverable. It is a numerics change, so not adoptable
mid-series, but it should be on the list afterwards.

*`micro_batch_seqs: 64` doubles the all-gather traffic.* Allgather launches per
**micro**-step are 36 in both dense (144 over grad_accum 4) and MoE (288 over
grad_accum 8), so FSDP re-gathers the whole model once per micro-step rather
than once per optimizer step, even with `reshard_after_forward: false`. Halving
the micro-batch to fit memory therefore doubled the parameter all-gather. This
is a direct, quantified cost of the OOM workaround, not of MoE.

**The idle is host-dispatch bound.** MoE issues 27,470 kernel launches per step
against dense's 7,057, 3.9x, of which grad_accum explains only 2x. The
elementwise/copy count is the worst offender at 9,000 launches for 141 ms, an
average of 15.7 us per kernel, against dense's 1,684 for 22 ms. Two independent
signatures confirm the diagnosis: the arm with the most GPU work per step
(g7-S12) has the *least* idle (6.3% against g4-S8's 13.8%), which is what a
fixed host cost being amortized looks like; and profiling itself, which adds
per-launch CPU work, slows the MoE arms 19% but dense only 2.3%.

**Caveat on the idle numbers.** The per-step idle read directly off the trace
(187-406 ms) is inflated by that same profiler CPU overhead, which is why the
table above derives overhead from steady-state `dt` minus measured compute
instead. The inflation is evidence for the diagnosis, not a number to quote.

**None of this biases the grid.** All four cells carry the same bs64, the same
grad_accum, and the same fp32 reduce dtype, so the within-grid comparison is
clean. It does matter for reading MoE against the dense control: part of the
wall-clock penalty MoE is being charged under wall-clock matching is the bs64
workaround rather than MoE itself. Worth stating in the paper rather than
quietly letting MoE wear it.

CUDA graphs are the obvious answer to a host-bound step and are already ruled
out here: per-block compile with `reduce-overhead` was a 10x regression under
FSDP2 and is not to be re-attempted.

### An all-NaN router row becomes an out-of-bounds index, and kills the job
`t2b-moe-g7-S12` (job 1774447) died at step ~5790 after 4 h 33 m of healthy
training: dt flat at 2776 ms, loss descending, `moe_dead=0`, `grad_norm=0.11`
at the last log 10 steps earlier. All 16 ranks aborted with SIGABRT via the
NCCL watchdog, which reads like a network fault and is not one.

The real error is a device-side assert in an inductor kernel:

    Assertion `index out of bounds: 0 <= tmp4 < 83` failed.

83 is `moe_num_experts`. A router top-k index came back equal to the expert
count. `_router_topk_kernel` (`moe.py:164`) picks the winning column with

    idx = tl.min(tl.where(eq, cols[None, :], E_), axis=1)

where `eq` is `x == m[:, None]`. `E_` is the "no column matched" sentinel and
`E_` *is* `num_experts`, so whenever no column compares equal to the row max the
kernel writes exactly the first out-of-bounds index, unguarded. The downstream
`logits.gather(-1, indices)` then asserts and takes the device down.

Reproduced on one GH200 at E=83, top_k 6:

| row content | result |
|---|---|
| normal | in bounds |
| one NaN among 83 | in bounds (Triton's `tl.max` skips NaN) |
| entirely +inf | in bounds |
| **entirely NaN** | **all six indices = 83, out of bounds** |

A single NaN is survivable; an entire NaN row is not. That is the diagnostic
detail, because `raw_logits = F.linear(x, w)` makes a whole row NaN exactly when
one token's *hidden state* carries a NaN: the dot product against all 83 expert
rows is NaN together. So the true event was one NaN hidden state, and the router
kernel converted a recoverable numerical event into an unrecoverable crash with
a misleading NCCL traceback.

**Correction (2026-09-17): do NOT "fix" the sentinel by clamping it.** The
original entry below called the sentinel a defect and proposed making the index
safe. That is wrong, and it took a second crash to see why. Clamping `E_` to a
valid index would make the NaN *silent*: the gather would return a
finite-but-meaningless routing weight, training would continue, and the
corruption would reach the weights on the very next optimizer step. The
out-of-bounds assert is the only reason we know this happens at all. The fix is
detection at the source, not suppression at the symptom.

It also reframes what a completed MoE run means. A NaN in any token's hidden
state, entering any of the 30 MoE routers, produces an all-NaN row and aborts
the job in the forward pass, before any optimizer step. So the sentinel is an
accidental activation-level NaN detector at 30 of 32 layers. **A MoE run that
finished carries a guarantee no dense run has: no NaN reached any MoE layer
input at any step.** That, rather than "the checkpoints are finite", is why the
scored results stand. (Verified anyway: all three saved MoE checkpoints are
fully finite, max |param| ~4, `correction_bias` bounded at 0.49-0.57.)

Two separate defects:

1. **The sentinel is unsafe as an index** but load-bearing as an alarm. See the
   correction above: leave it loud.
2. **Something produced a NaN hidden state** at ~step 5790 under a configuration
   whose three sister arms, same LR, same optimizer, ran to completion. Origin
   not established. Could be a rare numerical edge or a transient fault.

**Operationally: MoE arms have no crash recovery.** `save_steps` is 100000 and
the wall-clock stop is what writes the checkpoint, so a mid-run abort loses
everything. This one cost 4 h 33 m and produced no artifact. The obvious
mitigation, periodic saves, is not free: the wall-clock budget is
`time.perf_counter() - wallclock_t0` (`pure_pipeline.py:3477`) and does **not**
pause for checkpointing, so intermediate saves of a 24 GB checkpoint come
straight out of the matched 6.5 h and would handicap that arm against its
siblings. Relaunched identical (job 1777903) rather than break matching.

**It did fail twice** (`t2b-moe-g4-S12-re1`, job 1836251, 2026-09-16, step
16180 after 5 h 35 m, this time surfacing as
`ScatterGatherKernel.cu:163 idx_dim >= 0 && idx_dim < index_size` rather than
the inductor-fused gather).

**Error ordering, checked properly on 2026-09-22** (it had been asserted rather
than verified, and the second stderr also carried a flash-attention CUDA error
and a CUBLAS failure that would mean something quite different if either came
first):

| | g7-S12 | g4-S12-re1 |
|---|---|---|
| first error, line | 136 | 136 |
| signature | `index out of bounds: 0 <= tmp4 < 83` | `idx_dim >= 0 && idx_dim < index_size` |
| kernel | inductor-compiled gather | eager ATen ScatterGatherKernel |
| flash-attention error | line 6028 | line 504 |
| CUBLAS / watchdog | -- / 10671 | 836 / 1079 |

The gather assert is first in both, with only startup boilerplate before it, and
the flash-attention and CUBLAS entries appear thousands of lines later and are
themselves "device-side assert triggered", i.e. downstream of the poisoned
context. **So the NaN does not originate in attention.**

One difference stays unexplained: the same gather ran compiled in one crash and
eager in the other, which is why only g7-S12 printed its bound. That is why
g7-S12 is *proven* to be the router (bound 83 = `moe_num_experts`, gathering
from the stride-88 padded router logits) while g4-S12-re1 is only consistent
with it. Tonight's check A settles that by naming the layer directly. The earlier plan said that would mean redoing all
four arms against a fixed kernel. It does not, because the "fix" was wrong; see
the correction at the top of this section.

**The source is MoE-specific, not a cluster property.** The whole series was
grepped for the pipeline's non-finite-loss skip
(`"Skipping optimizer step %d due to non-finite loss"`, `pure_pipeline.py:3091`)
across 280 `.out` and 278 `.err` logs: **zero hits, ever, dense or MoE**. So the
only NaN evidence in the series is these two router crashes. Exposure:

| | runs | run-hours | NaN events |
|---|---|---|---|
| dense | 156 | 392.2 | 0 |
| MoE | 12 | 46.8 | 2 |

MoE's rate is one per 23.4 run-hours. At that rate dense's 392.2 hours would
have produced 16.8 events; observing zero has Poisson probability 5.3e-8. The
caveat worth stating: the two detectors differ. Dense only catches a NaN that
survives to the loss, MoE catches any NaN entering a router. Attention mixes
tokens, so a dense activation NaN should still reach the loss, but the
sensitivities are not identical.

Prime suspect is therefore the sonicmoe/quack path: it is the only MoE-specific
compute, it is version 0.1.2.post1 from a fork, and it is what dense does not
run. Not established, and a transient hardware fault is not excluded by this
data alone.

### MoE eval is 96% quack autotuning, and almost all of it is waste
The first MoE eval (`t2b-moe-g4-S12`, job 1782359) ran over 75 minutes without
finishing, against ~18 min per checkpoint for the dense arms. Sampled the live
process with py-spy: **24 of 25 stacks were inside `quack/autotuner.py`**, under
`check_disk_cache -> benchmark -> _bench_cuda_graph_l2_rotate`. Not model
forward, not the probe training. Kernel autotuning.

Three compounding causes.

**1. The eval launcher never set the quack cache location.** `sbatch_run.sh`
exports `QUACK_HOME` and `QUACK_CACHE_DIR` into `$FSROOT/cache/quack`, which has
7,203 entries the training runs paid for. `sbatch_eval.sh` exported neither, so
`default_cache_dir()` (`quack/autotuner.py:64`) fell back to
`Path.home()/.quack/cache`. That puts run artifacts off fscratch, against the
standing rule, and throws away every entry training had already computed.
Fixed: the two exports added to `sbatch_eval.sh`.

**2. The autotune key contains every tensor's shape.** `autotuner.py:613-620`
builds the key from the tuning keys plus `str(arg.shape)` for each tensor
argument, so the token count `M` is part of it. Eval batches variable-length
proteins under `dynamic=True`, so nearly every batch is a new key and a fresh
autotune. The cache does not converge: new entries arrived at 13 per 120 s at
69 min and 15 per 120 s at 73 min, still climbing.

**3. Autotuning buys nothing here anyway.** Measured on one GH200, the real
`MoELayer` sonicmoe path at E=47/top_k 3/inter 672, cold cache, five distinct
token counts:

| new shape | `tuned=True` | `tuned=False` |
|---|---|---|
| M=3137 (first, includes CuTe compile) | 65.7 s | 7.7 s |
| M=5211 | 10.9 s | 0.007 s |
| M=7307 | 11.0 s | 0.005 s |
| M=9401 | 11.5 s | 0.006 s |
| M=11503 | 11.5 s | 0.006 s |
| **total** | **110.5 s** | **7.8 s** |

Roughly 11 s per new shape, against 6 ms untuned. And the tuned kernel is not
faster in steady state: repeat-call latency was 2.48-3.26 ms tuned against
1.61-2.76 ms untuned. That comparison is single-sample at millisecond scale so
it does not establish that untuned is *faster*; what it does establish is that
there is no measurable win to pay 11 s a shape for.

`quack.gemm_interface.gemm_out` and friends take `tuned: bool = True`, and
`tuned=False` routes to `partial(gemm_tuned.fn, config=None)`, skipping the
search. sonicmoe calls `gemm_gated` etc. without passing it
(`sonicmoe/functional/__init__.py:109`), so there is no env knob; forcing it off
for eval needs a wrapper around the four `quack.gemm_interface` entry points.

**This is MoE-only.** Dense checkpoints never call a quack GEMM, which is why
every eval before this one was fast and why the problem appeared the moment the
first MoE checkpoint was scored.

**Fixed, and verified.** Two changes landed:

- `slurm/sbatch_eval.sh` now exports `QUACK_HOME` and `QUACK_CACHE_DIR` into
  `$FSROOT/cache/quack`, the same cache training uses.
- `eval/nanoplm-src-sync/src/nanoplm/eval/_quack_untuned.py` (new) rebinds the
  four `quack.gemm_interface` entry points to pass `tuned=False`, and rebinds
  them in `sonicmoe.functional` too, which binds them by value at import.
  `cli/eval.py` calls it at the top of `run()`, before the model is built.
  `NANOPLM_EVAL_QUACK_TUNED=1` restores the autotuner. The module lives in the
  eval-only tree; the frozen training tree is untouched and training still
  autotunes, which is correct there since it sees a small fixed set of shapes.

Re-ran as job 1782946. py-spy: **0 of 12 stacks in the autotuner**, against 24
of 25 before, and the fscratch quack cache stayed at 7,203 entries, so nothing
is being tuned at all.

Still open, and not needed now: bucketing eval batches to fixed token counts
would make the cache hit rather than be bypassed. Only worth it if a future eval
path wants tuned kernels.

One numerical note. `tuned=False` selects a fixed kernel config instead of a
per-shape one, so bf16 reduction order differs from a tuned run. Nothing already
scored is affected: dense checkpoints never call a quack GEMM, and no MoE
checkpoint had been scored before this. Every MoE arm will be evaluated under
the same setting.

### The NaN hunt: six runs launched 2026-09-22
Two crashes in 46.8 MoE run-hours against zero in 392 dense run-hours is a rare
event, so the design is parallel exposure plus an instrument that localises the
source, not one run with more logging.

**Why more `debug_layerwise_metrics` alone would not have answered it.** It was
already on (`debug_layerwise_log_every: 20`) during both crashes and recorded
nothing: zero non-finite entries, and activation magnitudes at the last record
before each crash (1.62e5 for g7-S12, 1.80e5 for g4-S12-re1) matching the clean
dense run's 1.86e5. Its hooks also sit on block *outputs* while the router sits
inside the block, so the crashing step never reaches them however often they
fire. Raising the cadence to 1 still buys something real, though, and is set on
all six arms: the last record before the g4-S12 crash was step 16180 and it died
in 16181-16200, so a ramp inside those steps was invisible. Per-step gives the
approach; the probe gives the crash itself.

**The probe** (`moe_nan_probe.py`, new, guarded by `NANOPLM_MOE_NAN_PROBE=1`, so
the frozen training tree is bit-identical when off). Two checks per MoE layer:

    check A  router input   -> NaN here means attn/norm/residual upstream
    check B  expert output  -> NaN here with A clean means the expert kernel

On a trip it dumps the offending tensors (plus `x`, `indices`, `weights` for B),
the layer index and the rank to `$NANOPLM_MOE_NAN_DUMP_DIR`, then raises naming
which side tripped. Validated on the login GPU by injecting a NaN into one
token's hidden state: clean forward does not trip, injected NaN trips check A at
the right layer and writes the dump.

**The discriminator is the arm split, not the dump.** Three arms on `sonicmoe`
and three on `cutlass`, all resuming the same `checkpoint-8778`, so the only
difference is the expert kernel. If only the sonicmoe arms crash, that is the
answer without reading a byte of the dump. `TORCH_USE_CUDA_DSA=1` on all six so
a device assert reports its launch site rather than the next sync.

| arm | job | backend |
|---|---|---|
| nan-sonic-1 | 1944350 | sonicmoe |
| nan-sonic-2 | 1944352 | sonicmoe |
| nan-sonic-3 | 1944354 | sonicmoe |
| nan-cutlass-1 | 1944351 | cutlass |
| nan-cutlass-2 | 1944353 | cutlass |
| nan-cutlass-3 | 1944355 | cutlass |

All six resume `t2b-moe-g4-S12-1774445/checkpoint-8778` (g4-S12, E=47, top_k 3,
inter 672, bs64, LR 2e-2), 6 h wall-clock each, `checkpoints-nan/<arm>/`,
dumps in `nan_dumps/<arm>/`. Step time 3.78-3.87 s (sonicmoe) and 4.07-4.16 s
(cutlass) against 2.64 s for the same arm without the per-step logging, so the
diagnostics cost about 45%. That is fine here and would not be in a matched arm.

Deliberately NOT varied: `micro_batch_seqs` stays 64 on every arm. Trying 128
would OOM and confound the comparison.

Prior: at one event per 23.4 run-hours, six 6 h arms give roughly a 75-80%
chance of at least one crash. A clean sweep is itself informative but weak.
### Auditing the six custom MoE Triton kernels: they are not the NaN
All six were LLM-written, so they were differential-tested against PyTorch
references on one GH200 rather than read.

| kernel | path | verdict |
|---|---|---|
| `_router_topk_kernel` | **shared** | 7 shape combos exact vs `torch.topk` on gathered values; correct lowest-index tie-break on all-equal rows; counts histogram exact |
| `_moe_gather_kernel` | cutlass | forward bit-exact vs `x[idx]` |
| `_moe_scatter_add_kernel` | cutlass | indices correct, but bf16 `tl.atomic_add` accumulation |
| `_moe_gather_combine_fwd` | cutlass | max diff 4.8e-7 |
| `_moe_gather_combine_bwd_fused` | cutlass | grad_expert exact, grad_weight 4.9e-4 |

**Two suspicions raised on reading and then disproved.** The `tl.float64`
accumulator branch is only selected when the input genuinely is float64
(`fp32_accum = dtype != torch.float64`), so it is correct. And
`_moe_autotune_key` is only used for status printing: Triton's real key is
`key=["C"]`, and every kernel loops `cdiv(C, BLOCK_D)` under a mask, so any
block size is valid for any C. There is no stale-config hazard of the kind the
canon kernels hit.

**The one real defect: bf16 atomic accumulation in the dispatch backward.**
`grad_x[token_idx[i]] += grad_sorted[i]` accumulates in bf16 where ATen's
`index_add` accumulates higher. Diagnosed rather than guessed: the error is
**exactly zero on rows receiving a single contribution** and non-zero only where
contributions collide, in both fp32 and bf16, which is accumulation order and
not indexing. Cost at bf16: p99 relative error 3.9%, max absolute 0.125. Fix is
`torch.index_add_`.

**Extreme-value probe, with a control.** Feeding finite-but-huge values does
make the combine and dispatch kernels emit non-finite output, but **the PyTorch
reference goes non-finite on the identical inputs**, so it is arithmetic
overflow rather than a kernel defect. It needs magnitudes near 3.4e38; the
largest activation the debug metrics have ever recorded in these runs is 1.9e5,
a factor of 1.8e33 below.

**What this does and does not license.** It raises confidence in our code and it
makes `_router_topk_kernel` a victim rather than a source, since it is correct
given finite input. It does **not** touch the prime suspect: both crashes ran
`sonicmoe`, whose expert compute is third-party quack/CuTe, untested here. The
four cutlass-path kernels were never on the crashing path at all.

**Standard replacements**, if the maintenance burden ever outweighs the fusion:
`index_select` for the gather, `index_add_` for the scatter-add (which also
fixes the precision), `(eo[inv] * w.unsqueeze(-1)).sum(1)` for the combine with
autograd supplying both backwards, `torch.topk` + `bincount` for the router (the
existing fallback already is this), and `torch._grouped_mm` for the whole
dispatch-plus-expert path, already measured 7-11% faster than the hand-built
CUTLASS extension with no dependency and no multi-rank JIT race. What the custom
versions buy: the counts histogram for free, and not materializing `eo[inv]`.

### The dense-vs-MoE NaN rate is confounded by batch size
Recorded earlier: 2 events in 46.8 MoE run-hours against 0 in 392 dense
run-hours, P(0) = 5.3e-8, concluded "MoE-specific". **That conclusion is not
supported as stated.** Every MoE run in this series uses
`micro_batch_seqs: 64`; every dense run uses 128. The two are perfectly
confounded, so the same numbers are equally consistent with a bs64-specific
cause: different grad_accum, different packing, different padding structure.

It is tempting to argue bs64 is cleared because the four LR-sweep arms and the
three completed grid arms give ~47k bs64 steps with no event. That argument is
wrong: those steps and the two crashes are the same population (all bs64, all
MoE), and zero-event samples from a population cannot refute a hypothesis that
predicts events in exactly that population. The rate is 2 per ~60k
bs64-and-MoE steps and the data attributes it to neither factor.

Separating them costs one run: a dense arm at bs64. Worth doing before any
conclusion about sonicmoe is written down.

### Ruling out LR, and what the crash data positions say
**LR is unlikely.** Over the resumed run that crashed, max weight norm grew
1.39x against the dense control's 1.34x at the same 2e-2, and max activation
ended at 1.80e5 against dense's 1.86e5 -- MoE converges toward the dense level
rather than diverging from it. `grad_norm` sat at 0.07 and loss descended
smoothly to the last logged step. There is no divergence signature. Dense also
runs 2e-2 for 392 run-hours with no event.

**Data position: different for the two crashes.** epoch_data_step ~46,320
(epoch 0) for g7-S12 and ~70,488 (epoch 3) for g4-S12-re1, so a single bad
corpus sequence is not indicated, though per-epoch shuffling means it is not
excluded either. The useful part is the prediction: the six diagnostic arms all
resume the same checkpoint and traverse identical data, so **a data cause makes
all six crash at the same step**. Nothing else in the hypothesis space does
that.

**Hypotheses still live**, in rough order: the quack/CuTe expert kernels
(untested here, third-party, only on the crashing path); something upstream in
attention/norm/residual; the bs64 padding path; and a hardware fault. Our own
six Triton kernels are audited and cleared, and the router top-k in particular
is a victim rather than a source.

### The NaN caught in the act: it is not sonicmoe, and not padding
Six diagnostic arms, 2026-09-22. Five completed 6 h clean. **`nan-cutlass-1`
tripped the probe at step 9799**, 1 h 24 m in.

| arm | result |
|---|---|
| nan-sonic-1 / 2 / 3 | COMPLETED 6h11m, step ~14,400 |
| nan-cutlass-2 / 3 | COMPLETED 6h14m, step ~13,930 |
| **nan-cutlass-1** | **FAILED, probe trip at step 9799** |

**sonicmoe is exonerated.** Three sonicmoe arms ran 18 h without an event and
the one that fired was `cutlass`. With the two earlier crashes (both sonicmoe)
that makes the NaN backend-independent, which retires the hypothesis this hunt
was designed around. The quack/CuTe kernels are not the cause.

**What the trip says.** Check A fired: the **router input** at **MoE layer 5**,
on all 16 ranks, with layers 2, 3 and 4 clean in the same forward. So the NaN
is made upstream of the expert path, in layer 4's MLP, layer 4's attention or
layer 5's attention. Layer 5 is `sliding_attention`, window 128.

**What the dump says**, and it kills two more hypotheses:

- **33,143,808 of 33,554,432 elements non-finite (98.8%)**. Not one token going
  bad: essentially the whole activation.
- **Rows 0-32555 are all NaN; rows 32556-32767 are clean.** The clean tail is
  the padding segment. So **padding survives and every real token dies**, the
  exact opposite of the padding hypothesis, which is dead.
- **Zero infs, all NaN.** Not an overflow path.
- No ramp: at step 9798 every weight and grad norm is finite and the max weight
  norm is growing smoothly (2188.7 -> 2189 over the preceding steps).

**The crux, still unexplained.** Whatever did this spared an independent
`cu_seqlens` segment while destroying every real sequence at once. A single NaN
token spreading through a 128-wide sliding window reaches its neighbours, not
32,556 tokens across 64 packed sequences. A NaN weight in any shared operation
would have taken the padding with it.

### Probe v2: no per-layer syncs, and a finer localisation
v1 blocked on `bool(isfinite(x).all())` once per layer, 240 syncs per optimizer
step, about +26% step time. It synced because the check ran before the router
purely to pre-empt the out-of-bounds assert.

v2 removes the reason rather than the symptom. With the probe on, the router's
top-k indices are clamped into range (out-of-place: the gather saves them for
backward), so the assert cannot fire and the forward always completes. Each
check then writes a verdict into a device tensor with no sync, and the last MoE
layer calls `finalize()`, which syncs **once per micro-step** and reads every
verdict at once. Clamping is safe because a tripped probe aborts at the end of
that same forward, so the garbage it produces never reaches an optimizer step.

It also checks **two** points per layer now, router input and layer output, so a
trip separates "this block's attention/norm" from "this block's expert path"
instead of leaving a two-layer window, and it dumps `cu_seqlens` and
`valid_token_count` for the packing question.

Validated by injection: a clean forward does not trip; a NaN injected into
layer 5's input reports `FIRST non-finite: router_input at MoE layer 5` plus the
full propagation chain. Microbenchmark shows +13% on a *one-layer* model where
the single sync is amortised over one layer instead of thirty; the real-model
cost projects to roughly 2%.

### Next: the test that separates MoE from bs64
Every MoE run in this series is bs64 and no dense run has ever been, so the two
remain perfectly confounded and this is now the load-bearing question. Six arms
queued (jobs 1950303-1950311), all bs64, all with
`debug_non_finite_params: true`:

- **3 x dense at bs64**, resuming `checkpoint-21448`. Dense has no `MoELayer` so
  the probe does not apply, and does not need to: the existing non-finite-loss
  tripwire logs and skips rather than crashing.
- **3 x MoE with probe v2**, resuming `checkpoint-8778`.

A trip on a dense arm exonerates MoE entirely and makes this an
attention/packing bug.

### Probe v2.1 fell back to eager; v2.2 fixes it and makes the NaN survivable
Two lessons about instrumenting a compiled model, both learned the expensive way.

**v2.0 OOMed.** It held a reference to every layer's router input and output
for the dump: 30 layers x 2 x 64 MB, ~3.8 GB pinned across the forward, against
an S12 arm that already runs at ~97.5% of the card. All three MoE arms died on
startup. v2.1 stored 32 KB per-row masks instead (1.9 MiB total, measured).

**v2.1 ran at +35%, not the ~2% projected.** The projection came from an *eager*
microbenchmark, which structurally cannot see dynamo recompiles. In training the
model is compiled whole, and `debug_layerwise_metrics` puts a
`torch.compiler.disable`d hook on every block, so each MoE layer's forward runs
as its own frame sharing one code object. v2.1 read per-instance Python ints
(`_probe_layer_idx`, first/last flags) inside that forward, so dynamo
specialised per layer, hit `recompile_limit (8)` on all 16 ranks, and fell back
to eager. Dense arms, with no MoELayer, logged zero such warnings.

**v2.2 keeps the traced region free of Python state:**

- Each MoELayer owns a non-persistent `_probe_stats` buffer (not in the
  state_dict, so checkpoints are unchanged, and only registered when the probe
  is on), written by pure tensor ops: per-row finiteness, first/last bad row,
  bad-row count, `valid_token_count`, inf count. It latches the first bad
  micro-step of the accumulation window.
- The router index clamp stays, so a NaN forward yields a NaN *loss* instead of
  a device assert.
- That loss is caught by the pipeline's existing non-finite-loss skip, outside
  compiled code, which calls `dump_from_model()` and resets the buffers. **The
  event is now survivable**: the step is skipped, the stats are written, training
  continues, so one run can record several events rather than dying on the first.
- Safety guard: with the clamp, a NaN forward no longer crashes, so it would
  reach `optimizer.step()` unless the skip path catches it. The pipeline now
  refuses to start with the probe on and `debug_non_finite_params` off.

**Validated under `torch.compile` with a compile-disabled hook per layer**, the
same structure as training, which is the test v2.1 should have had:

| mode | ms/iter | unique graphs |
|---|---|---|
| probe off | 54.57 | 1 |
| control: per-instance int read in forward | 57.00 | **6** |
| **v2.2** | 56.90 | **1** |

The control proves the harness detects the specialisation (one graph per
layer); v2.2 keeps one shared graph at +4.3%. An injected NaN gives a non-finite
loss with no device assert, and the dump reports
`FIRST non-finite at layer 5 router_in: bad rows 12000/32768 [0..11999]
contiguous=True, n_inf=0`, exactly the injection. Reset clears it.

Relaunched as jobs 1959381-83 (three MoE, v2.2) alongside the three dense-bs64
arms already running (1950303/07/10). The cancelled v2.1 arms had reached steps
9700-9720 with no trip.

### The NaN is not in the data, not one node, and not NorMuon's per-expert step
Three results from the v2.2 batch (jobs 1959381-83, all COMPLETED 6 h, no
events; at one event per ~37k MoE steps, 20.7k steps had a 57% chance of none).

**Not a function of the data.** Nine arms resumed the same `checkpoint-8778`
with the same data order. Eight passed step 9799 cleanly (sonic-1/2/3,
cutlass-2/3, n2-moe-1/2/3); only `nan-cutlass-1` died there. A bad sequence would
have hit all of them. Trajectories do diverge after the resume through
nondeterministic kernels, so this rules out a *deterministic* data trigger, not a
data-plus-weight-state one.

**Not one bad node.** The three crashes ran on disjoint node sets:
`jpbo-081-[02,05-07]`, `jpbo-048-[19,21,23-24]`, `jpbo-047-[03,08-09,15]`.

**A hypothesis that fits every feature of the cutlass-1 dump.** Suppose one
expert's *output* at layer 4 goes non-finite. Only the tokens routed to it (top-3
of 47, ~6%) are hit. Layer 5 is sliding attention, window 128, so within each
sequence those tokens contaminate their neighbours until almost every real token
is NaN: the observed 98.8%. The padding segment is 212 identical tokens, which
route identically and form their own attention segment, so it stays clean.
Weights are all-gathered, so all 16 ranks trip together. Layers 2-4 router
inputs are clean because layer 4's router runs before its experts. And if the
expert output overflowed to inf, layer 5's RMSNorm turns inf/inf into NaN, which
matches zero infs at the router input. It is backend-independent and MoE-only,
since only experts see sparse, step-to-step-varying token sets.

**The optimizer is not the obvious source.** The natural way for one expert to
go bad is a NaN written into its weights by the update. Tested `dion.NorMuon`
directly on stacked (47, 672, 1024) expert tensors at the training settings
(lr 2e-2, mu 0.95, beta2 0.95, wd 1e-5, spectral-norm lr adjust, eps 1e-8). All
finite under: normal gradients; one expert with exactly zero gradient every step;
an expert alive, dormant for 40 steps, then woken by a 1e-12 gradient; a ~1e-33
near-underflow gradient; one dormant row inside an expert; and only one of 47
experts receiving any gradient. Reading the code agrees: the per-row step updates
the variance with the current update *before* dividing, which bounds a row at
`1/sqrt(1-beta2)`, and both divisions carry a 1e-8 floor. Caveat: this is
single-GPU; the FSDP-sharded megabatch path was not tested.

**What the v2.2 probe will say if it fires again.** The hypothesis predicts
layer 4 `layer_out` bad with ~6% of rows, *non-contiguous*, and layer 4
`router_in` clean; `n_inf > 0` there would mean the expert overflowed and RMSNorm
did the rest. `debug_non_finite_params` is now on, so the skip path will also
report whether any weight was non-finite at that step, which settles the
optimizer question in the sharded setting too.

### No cross-stream race, and the compiled position path is correct
**The v2.x arms have seen no event in 64.9k MoE steps**, against three in the
88.3k steps run without probe or with v1. At the earlier rate (one per ~29k)
that has an 11% chance of happening by luck, so something in the v2.x setup
might be suppressing the bug. The natural suppressor is a timing race: the v2.x
arms add work per layer (probe) and async host copies per step
(`debug_non_finite_params`), and a race is also nondeterministic, node-agnostic
and MoE-only. The candidate mechanism was FSDP-gathered expert weights being
read by a custom kernel (quack or the CUTLASS extension) before the gather's
stream sync.

**PyTorch's CUDA stream sanitizer finds no race.** Full MoE model, FSDP on one
node (4 GPUs), eager so every op is visible, 4 steps, in all three
configurations: no probe (`debug_non_finite_params` off, mirroring the grid
arms), v1 (off, mirroring the first hunt) and v2.2 (on, mirroring n2/n3). No
report in any of them (jobs 1966139/42/44).

The negative is trustworthy for three reasons: a positive control in the same
environment (a deliberate unsynchronised cross-stream write) is flagged; the
sanitizer is armed from the env var at `torch/__init__.py:2841`; and the runs
took 21-24 s per step, which is what instrumenting every op costs. And because
the sanitizer checks the synchronisation pattern rather than waiting for a NaN,
a structural missing sync would be flagged on the first step it executes, so
four steps is sufficient.

Coverage: every custom-op dispatch (router, sonicmoe, quack) and FSDP's own
stream handoffs. The profiler trace puts all MoE compute on the main stream (7);
the other streams are NCCL all-gather (24), reduce-scatter (28) and FSDP copies,
so there is no private stream for an expert kernel to race on. Blind spots:
inside NCCL (shared with dense, which never NaNs), the 16-GPU multi-node
topology, and the compiled path, since the sanitizer cannot see inside Inductor
kernels.

Setup note: the sanitizer hooks tensor ops process-wide, including in forked
DataLoader workers, where touching CUDA is illegal ("Cannot re-initialize CUDA
in forked subprocess"). It needs `num_workers: 0`. `TORCHDYNAMO_DISABLE=1` gives
an eager run without code changes.

**The compiled position path is correct.** `_position_ids_from_cu_seqlens` uses
`repeat_interleave(seq_lens)` with no `output_size`, a data-dependent shape that
Inductor lowers to cumsum + searchsorted, the same family as the canon-layer
codegen bug. Fuzzed compiled against eager on 3,000 realistic packings (~64
sequences of 20-512 tokens plus a padding segment, 32,768 total), under both
`dynamic=False` and `dynamic=True`: 0 disagreements.

**What is left.** Bad luck (11%); a compiled-path bug somewhere other than the
position computation; NCCL internals or 16-GPU topology; hardware faults.

### TE's sync-free grouped GEMM: Blackwell-only in 2.15, Hopper from 2.16/2.17
`nvte_grouped_gemm` and its two variants take device-side group metadata, which
is exactly what a MoE dispatch wants. In the TE we run they are unusable:

    common/gemm/cublaslt_grouped_gemm.cu:304
    NVTE_CHECK(cuda::sm_arch(current_device) >= 100,
               " requires Blackwell (SM100) or newer architecture.");

That is a hard runtime check, not a compile guard. GH200 is sm_90, so no CUDA
version reaches it. The cuBLAS 13.2 requirement in the header is a second
condition, not the binding one.

TE's other Hopper-capable paths need host-side shapes:

- `GroupedLinear.forward(inp, m_splits: List[int])` does `len`, `sum` and
  `torch.split` on a Python list.
- TE's own SM90 CUTLASS kernel (`gemm/cutlass_grouped_gemm.cuh:279`) builds
  `problem_sizes_host` in a loop over separate per-expert tensors.

So adopting TE on Hopper would *add* a device-to-host sync per MoE layer.

Our own cutlass backend already does this: `batch_sizes` must be a CUDA tensor
(`csrc/moe_cutlass_grouped_gemm.cu:419`) and the problem shapes are built by a
device kernel (`:253-277`). Sync-free. So is `torch._grouped_mm`, which is in
our torch and needs nothing installed.

**Upstream fixed the Hopper gap, in two steps.** The paragraphs above describe
TE 2.15 and 2.16, which is what we run. Newer releases:

- **2.16**: the C-level check became `sm >= 90`, so the kernel supports Hopper,
  but PyTorch `GroupedLinear.forward` still took `m_splits: List[int]`, so the
  host sync was still there one level up.
- **2.17**: `GroupedLinear.forward(inp, m_splits: torch.Tensor)`, offsets built
  on device by `tex.splits_to_offsets`. Genuinely sync-free on Hopper.

It needs **cuBLAS 13.4+** on Hopper (13.3 elsewhere, 13.5 for fp8 per-tensor
current scaling) and `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1`, which
defaults to 0. We have cuBLAS 13.1.1. fp8 *delayed* scaling is not supported on
that path, only current and block, and delayed is the one that won our fp8
grouped-GEMM benchmark.

The 2.18 fused grouped MLP (GroupedLinear + activation + GroupedLinear) does
not help: `fuse_grouped_mlp_ops` returns unfused unless the recipe is mxfp8 or
nvfp4, and the CuTeDSL variant gates on compute capability 10. Blackwell only.

### TE 2.18 on Hopper: built, measured, tied with torch._grouped_mm
We built TE 2.18 against a CUDA 13.4 toolkit and ran it. The Hopper grouped
GEMM works: `_is_grouped_tensor_path_supported` returns True at cublasLt
130401 on cc (9,0), forward and backward, with `m_splits` as a device tensor.
Output matches `torch._grouped_mm` exactly.

It is also 0.4% to 1.7% *slower* than `torch._grouped_mm`, which we already
have for free. Both sit 2.0x behind sonicmoe, because both need the same
materialized scatter and gather around the GEMM.

**It caps at 64 groups.** `cublaslt_grouped_gemm.cu:667`:
`A_list supports up to 64 tensors per kernel, got 95`. The g8-S12 cell (E=95)
cannot run on TE at all. Nothing else we tested has that limit.

Two traps worth writing down, because both fail quietly or misleadingly:

1. A runtime cuBLAS upgrade is not enough. The kernel is behind
   `#if CUBLAS_VERSION >= 130300`, so TE must be *compiled* against 13.3+
   headers or it raises `compile-time cuBLAS version is 130000` at the first
   call, while the Python-level gate still says True.
2. nvcc prefers its own bundled headers over `CPATH`, so you cannot shim
   newer cuBLAS headers into an older toolkit. You need a whole newer
   toolkit. The newest `nvidia-cuda-nvcc` wheel is 13.4.46rc1, which is
   exactly the Hopper bf16 threshold and one short of the 13.5 that fp8
   per-tensor current scaling on Hopper wants.

## Cluster and launcher

### Slurm 25.05.9 broke three things at once (2026-09-08 maintenance)
The maintenance installed `slurm 25.05.9-1.20260903git791df5b`. Every 4-node job
failed afterwards. Three independent regressions, all cluster-side, isolated
with jobs 1721889 / 1721915 / 1721921 / 1721925:

**1. The first `srun` in a fresh allocation always fails.**

    PSI: doSpawn: spawn to node N failed: ""
    Could not spawn 'bash' process 0: Invalid argument
    spawnSingleExecutable: PSI_spawnRsrvtn() failed

Position, not flags. A flagged srun fails in first position and a flagless one
succeeds in second, so it is not about `--cpu-bind` or `--distribution`.
Workaround: burn a throwaway `srun ... true || true` before the real one.

**2. The nvidia-smi sampler shape fails in any position.**
`--overlap --ntasks-per-node=1 --ntasks=<nodes>` inside an
`--ntasks-per-node=4` allocation, with or without flags. Removed the sampler.

**3. One GPU is now bound per task.** Each task gets
`CUDA_VISIBLE_DEVICES=<localid>`, so the only valid ordinal inside a task is 0
and `torch.cuda.set_device(LOCAL_RANK)` raises "invalid device ordinal" on
local ranks 1-3. The devices are only hidden by the env var, not by cgroups
(`nvidia-smi -L` still lists four), so `unset CUDA_VISIBLE_DEVICES` in the srun
wrapper restores the previous behaviour. Verified: device_count 4 and a real
allocation on `cuda:LOCAL_RANK` from all four tasks.

Forcing `LOCAL_RANK=0` also fixes CUDA and is WRONG: `utils.py` computes
`is_local_main = (LOCAL_RANK == 0)`, so every rank on a node would think it is
the local main.

Throughput after the fixes is unchanged: 501 ms / 51.8% MFU against 511 ms /
50.8% before the maintenance.

**Debugging lesson.** "RUNNING" in `squeue` is not evidence a job is healthy;
these died about 40 s in. And a `bash -x` launcher echoes its own comments into
stderr, so grepping stderr for an error string can match the comment that
documents it. Both cost a wrong conclusion tonight.

### A "retrying" log line that never retried

`tools/submit_gated.sh` waits for a job's profiler trace, and on the way it
watches for the job dying. When it saw a dead job it logged "died early ...,
retrying" and broke out of the wait loop with no trace in hand. The next branch
is `if [ -z "$t" ]`, "no trace -> keeping it, cannot gate", which returns. So the
retry never happened, and the log said it did.

Cost: two arms of the t2p LR sweep on 2026-09-11. Slurm killed both
(`CANCELLED by 0`, uid 0, node failure on jpbo-013) about 100 s in, and nobody
noticed until the other three finished two hours later and the results table had
three rows instead of five. They were the two most important learning rates.

Fixed by separating the two cases: a job that died gets `try=$((try+1));
continue`, a job that is merely slow to write a trace gets the old
keep-it-ungated behaviour. The fix also has to tell apart **who** cancelled it.
`sacct` reports `CANCELLED by <uid>`, and uid 0 is Slurm itself (node failure,
preemption), which deserves a resubmit. Any other uid is a person, and
resubmitting a job someone just cancelled is the last thing they want: that same
afternoon a deliberate cancel of five jobs would have fought a script trying to
put them back.

Two general lessons. A log line that claims an action must be written next to
the code that takes it, or it becomes a lie during a refactor. And when a batch
of N jobs comes back with fewer than N results, count them before reading them.

### Held jobs do not start themselves after maintenance
The Tier 1 jobs were submitted just before a cluster-wide maintenance window
and ended up `PENDING` with `Reason=JobHeldUser` and `Priority=0`, which is a
hold, not a queue position. A held job stays held after the reservation ends.

So a submission that straddles a maintenance window needs an explicit
`scontrol release <jobids>` afterwards. `squeue` showing PENDING is not enough
to conclude a job will eventually run: check the Reason field.

### sbatch had no --nodes default
A job silently ran on 1 node instead of 4. The pipeline compensated by raising
grad_accum to 4, so it completed normally and reported 53.88% MFU, which was
not comparable to anything. Nothing in the log said "1 node".

Fixed with `#SBATCH --nodes=4`. Same class of bug as `num_workers: auto`.

### The nvidia-smi sampler starved the training step
A plain `srun` step holds the node's resources, so the training step sat in
"step creation temporarily disabled" for 19 to 32 minutes. Needs
`srun --overlap`.

### The VRAM log line was misread
`peak=67,950/69,024MB` is peak allocated over peak reserved, both from the
caching allocator. It is not used-over-total. The card is 97,280 MiB. The log
now prints the card total too.

### A wall-clock stop does not write "checkpoint-stable-end"
It writes `checkpoint-<step>` and says so:

    Stopped on the wall-clock budget at step 920/100000; skipping the
    terminal 'stable-end' checkpoint. Latest state is checkpoint-920.

The terminal role name is reserved for a run that reaches the end of its LR
schedule. A wall-clock stop never does, and naming a mid-schedule snapshot
"final" would misrepresent it and could clobber a real one in the same output
directory.

Harmless once you know: the checkpoint is complete and
`ResumeConfig.checkpoint_dir` takes an explicit path.

## Working alongside another agent

### Two agent sessions will collide, and documentation does not stop it
Two Claude sessions forked from the same context are not independent workers:
they reach the same next action within minutes. Observed on 2026-09-09:

- 02:14 both submitted low-LR fill-in grids 34 seconds apart. Three of twelve
  runs were exact duplicates, ~67 GPU-hours.
- 10:32 one cancelled the other's ten 6 h Wave 2 runs five minutes in and
  replaced them with an equivalent set under different names.
- It also overwrote a shared tracking file, `run/tier1_wave2_jobids.txt`.

Asking the peer to stand down did not work; it did the opposite.

**Finding the other session.** Its processes and transcripts are invisible from
your node, because the cluster has many login nodes and an ssh lands on a
random one. Slurm records the submitter:

    scontrol show job <id> | grep AllocNode
    # AllocNode:Sid=jpbl-s01-01-interconnect-1.jupiter.internal:1621341

That gives the login node and the session id. In an open Claude window,
`!hostname` identifies which one you are looking at. Your own session is not
pinned either: this one submitted from jpbl-s01-02 and later reported
jpbl-s02-04.

**What actually prevents it.** A hard failure, not a convention:

- `OWNER.lock` in the series root names the owning session.
- `tools/make_configs.py` and `tools/abl_submit.sh` exit non-zero unless
  `$ABL_OWNER` matches. A fresh agent has it unset, so it fails loudly and has
  to ask rather than colliding silently.
- Tracking files are written as `run/jobids.<owner>.txt`, not a shared name.
- Rule added to AGENTS.md, which CLAUDE.md tells every agent to read.

The duplicates did have one accidental benefit: three same-config pairs gave
sigma_repeat ~0.0004 for free, which Tier 0 would otherwise have spent runs on.

## Terminal checkpoints did not record the data position

A checkpoint written by the periodic save carries `epoch_data_step`,
`epoch_sub_batch_step` and `batch_split_divisor`, and `resume.mode: continue`
uses them to fast-forward the loader to where the stream stood. The save that
runs after the training loop exits, which is the one every fixed-step arm ends
on, did not: those three fields were computed inside the loop and were not in
scope afterwards. A resume from a terminal checkpoint therefore started the
current pass from the top and silently re-trained on data the run had already
seen.

It surfaced because Tier 2c resumes every long run from its sweep winner's
terminal checkpoint, which is the exact case the bug covers. Fixed in
[PR #154](https://github.com/peymanvahidi/nanoplm/pull/154) by tracking the
position at each step boundary so the post-loop save can record it too, with a
guard for the zero-step case where no step ever completed.

Proven on three smoke runs rather than by reading: a periodic checkpoint carried
`epoch_data_step: 640`, a resume from it logged `fast-forward to epoch=0
data_step=640`, resumed loss at step 60 was 2.8024 against 2.8032 for the
continuous run (a 0.0008 gap where run-to-run nondeterminism alone is 0.0017),
and the patched terminal save carried `epoch_data_step: 960` = 60 x 16,
consistent with the 640 = 40 x 16 above.

The pinned tree was built before this and was patched in place on 2026-09-11
(`env/PINS.txt` records it, pre-patch file kept beside it). The patch only
populates fields in the saved `training_state.json` and does not touch the
training math, so arms either side of it stay comparable. One checkpoint was
already written without the position: the 2e-2 modernization sweep winner, whose
long run therefore re-sees 96 steps of epoch-2 data. Logged in
[plan.md](plan.md#now-the-modernized-base).

## A resumed run writes into the source run's directory unless you move ckp_dir

`utils.py` derives the run directory from the **checkpoint's name, not its
location**: `original_run_name = checkpoint_path.parent.name`, then
`run_root = ckp_root / run_name`. So a resume with `pretraining.ckp_dir`
unchanged reopens the source arm's directory and writes into it. The
`pretraining.run_name` in the resuming config is ignored for pathing; it only
shows up as a `-reN` suffix on the W&B run.

That is deliberate, and the code says so ("so it does not exist yet when
resuming into a fresh ckp_dir -- which is what you do to keep the source run
read-only"). The intended pattern is to point `ckp_dir` somewhere new, and then
the resumed run lands at `<new ckp_dir>/<original run name>/` with the source
untouched.

Missed on the first resume, `t2-mod-long` from the 2e-2 sweep winner. Two
consequences, both artifact-level rather than scientific:

- The gate cannot gate a resumed run. It waits for
  `checkpoints/<config name>-<jobid>*/profiler_traces/chrome_trace.json`, which
  never appears because the trace goes to the source arm's directory. The
  patched gate logged "no trace after 1500s -> keeping it, cannot gate" and left
  the job alone, which is the right fallback, but the run is ungated.
- The terminal save writes `checkpoint-stable-end` again, at the resumed run's
  final step, over the step-5000 checkpoint it resumed from. The sweep winner's
  trace was already overwritten before this was noticed; the checkpoint was
  copied out to `checkpoints/_preserved/t2p-mod-lr2e-2-step5000` first, verified
  equal by `training_state.json`.

Kept outside the run directory on purpose: a `checkpoint-`prefixed backup inside
it would be visible to checkpoint-listing and archiving code.

So every later resumed arm gets a fresh `ckp_dir`, and the sweep checkpoint it
resumes from is preserved before launch either way.

## The modernized base decouples the two contact readouts

The modernized base (rmsnorm + swiglu + QK norm, 15% 80/10/10, NorMuon 2e-2)
against stock ModernBERT at identical masking (`t1c-mlm15-m80`), both
wall-clock matched to 6 h, both scored native dev mode on the same eval tree,
steps 11039 and 10977:

| metric (dev) | stock | modernized | delta |
|---|---|---|---|
| eval loss @10500 | 2.2014 | 2.1751 | **-0.0263** |
| PGYM total SCC | 0.3600 | 0.3794 | **+0.0194** |
| zero-shot contact, sel long P@L | 0.4209 | 0.4048 | **-0.0161** |
| zero-shot contact, casp14 / casp15 long P@L | 0.1656 / 0.1928 | 0.1509 / 0.1691 | -0.0147 / -0.0238 |
| supervised contact, sel long P@L | 0.4162 | 0.4533 | **+0.0371** |
| supervised contact, casp14 / casp15 long P@L | 0.1871 / 0.2092 | 0.2092 / 0.2330 | +0.0222 / +0.0238 |

`sigma_seed` on the deciding metric is 0.0020 over the six base replicates, so
both contact results are past 5 sigma however the denominator is taken, and
every dataset moves the same way within each readout. **The two readouts
disagree in opposite directions**, which no previous arm in this series has
done:

| arm | zero-shot | supervised | probe - Jacobian |
|---|---|---|---|
| t0-rep1 (stock, 20% mask) | 0.3950 | 0.3941 | -0.0009 |
| t1c-mlm15-m80 (stock, 15%) | 0.4209 | 0.4162 | -0.0047 |
| t2-mod-long (modernized) | 0.4048 | **0.4533** | **+0.0485** |

On both stock architectures the two readouts agree to within noise. On the
modernized one the probe beats the Jacobian by 0.0485, and its probe score is
the best of any arm measured. So the contact structure is not lost: a trained
probe on the attention maps recovers *more* of it than from either stock model.
What degrades is the model's own output-sensitivity readout, which is what the
categorical Jacobian measures.

**Leading hypothesis, not yet a finding.** Of the three bundled changes only QK
norm touches attention, and both contact metrics read attention. The
implementation is parameter-free `F.rms_norm(q, (head_dim,))` with no learnable
gain, so `|q| = |k| = sqrt(head_dim)` exactly and the logits are capped at
+-sqrt(head_dim) after the `1/sqrt(head_dim)` scale. That is a hard ceiling on
how sharp any head can become, which would dampen a perturbation-based readout
while leaving the attention pattern's *relative* structure intact. It fits the
whole pattern: loss and PGYM fine, probe better, Jacobian worse. The test is one
arm of `base-modern15` with `use_qk_norm: false`; the candidate fix if it
confirms is a learnable per-head scale, as in OLMo-2 and ViT-22B.

**Confounds, stated rather than resolved.** The comparison bundles the three
architecture changes with the LR (7e-3 -> 2e-2), which is the only other config
difference (everything else including all four eval-masking keys and
`eval_mask_seed` is identical, so the losses are scored on the same masked
positions). A diagnostic on three sweep checkpoints at step 5000, same
architecture, LR the only variable, came back **non-monotone**: 7e-3 0.3662,
2e-2 0.3123, 2.8e-2 0.3531 on sel long P@L. With no downstream sigma at 5000
steps that neither implicates nor exonerates the LR; what it does show is that
the zero-shot readout swings by 0.054 across LRs of one architecture, which is
itself consistent with that readout being the fragile measurement.

**Loss did not predict any of this.** Across the 38 arms with both numbers, sel
long P@L regressed on eval loss gives r = -0.237, r^2 = 0.056. Among the 20 arms
within loss 2.19-2.21 the metric spans 0.0312, 7.6x the noise floor. Loss and
contact are close to independent axes in this series, so the -0.0263 loss win
never carried a prediction about contact either way.

Method note: the control had no provenance sidecar, so which checkpoint produced
its published 0.4207 was an inference from the single checkpoint on disk. The
re-score confirms it, reproducing 0.4205 -> 0.4209 (+0.0004), and incidentally
re-confirms that full-mode-then-dev-aggregate and native dev mode agree.

## Rope theta: a clean negative, and the Jacobian readout is the unstable one

Six arms off the modernized base, FSS pattern and LR 2e-2 held fixed, fixed-step
to 10500, one rule setting both thetas (longest wavelength = M x the largest
relative offset that layer type can use: 512 global, 64 local). Dev mode, all on
one eval tree. The anchor is the base theta at step 11039, a 5% step advantage,
so the ladder is read among the arms first.

| arm | global / local | loss | zero-shot long P@L | supervised long P@L | PGYM scc |
|---|---|---|---|---|---|
| anchor | 160k / 10k | 2.1751 | 0.4048 | 0.4533 | **0.3794** |
| A | 10k / 10k | 2.1753 | 0.4072 | 0.4538 | 0.3713 |
| B | 10k / 1200 | 2.1748 | **0.4283** | **0.4560** | 0.3694 |
| C | 2000 / 250 | **2.1744** | 0.3713 | 0.4468 | 0.3729 |
| D | 500 / 60 | 2.1749 | 0.3926 | 0.4387 | 0.3595 |
| E | 100 / 12 | 2.1818 | 0.3515 | 0.3994 | 0.3587 |
| F | 500 / 10k | 2.1772 | 0.4149 | 0.4376 | 0.3583 |

**Nothing here is worth adopting.** On the single-M ladder (anchor, B, C, D, E)
supervised contact and PGYM both decline smoothly as theta falls: Spearman
against log M of 0.900 for each, with zig-zag (mean size of the steps that go
against the trend) of 0.0007 and 0.0009. Two independent readouts agreeing on a
smooth monotone curve is the strongest signal in this table, and it says the
inherited theta is fine and lowering it costs. By M=1 the cost is unambiguous:
worst on all four measures. So the hypothesis that motivated this ablation, that
160k wastes most frequency pairs on a 512-token context, is **not supported**:
the near-constant pairs are apparently earning their keep as implicit NoPE
dimensions rather than going to waste.

**Loss is blind to theta.** Range 0.0074 across all six, and that is entirely E
and F; the other five span 0.0009, under half the resolution. C is nominally the
best loss of any arm ever run here (2.1744) and is fourth of six on supervised
contact. Anyone selecting a theta on loss would have picked almost at random.

**The zero-shot readout is the unstable measurement, not the signal.** Its
zig-zag is 0.0112, sixteen times the other two readouts, on the same five arms,
while its Spearman (0.800) looks respectable. Concretely: B beats C by 0.0570
and then C loses to D by 0.0213 going *further* down the ladder, which no smooth
relationship produces. Two other observations line up with it: the step-5000 LR
diagnostic on this architecture was also non-monotone with a 0.054 spread, and
this architecture is the one where the probe and the Jacobian
[came apart by 0.0485](#the-modernized-base-decouples-the-two-contact-readouts).
The supervised and PGYM curves being smooth on the same checkpoints rules out
"this architecture just has noisy downstream metrics" -- it is specific to the
Jacobian. That matters for the deciding-metric question, because the
pre-registered decider is the readout with sixteen times the jitter.

B's zero-shot win (+0.0235 on the anchor) is therefore not banked: it is the
largest number in the least trustworthy column, and B is +0.0027 on supervised
(nothing) and -0.0100 on PGYM.

**The local theta, on loss.** Lowering global alone is worse than lowering both:
F is +0.0021 on the anchor while D is -0.0002, and at fixed global 500 adding
the local change recovers 0.0023 (F 2.1772 -> D 2.1749). Small but consistent in
sign, and it is the thing the original single-knob A7 arm would have measured as
"theta does not help" for the wrong reason. On the downstream readouts the local
effect has no consistent sign, so loss is the only place it resolves.

## The sliding window buys ~1% of step time, and a 3.6% step-time gap was the network

All-global attention (`attn_layer_pattern: F`, 32 full layers) first measured
3.6% slower per step than the FSS base (2.015 vs 1.944 s/step), which would have
been charged to it under wall-clock matching. Comparing its profiler trace
against arm B's shows that is not what happened. Both traces carry an identical
49429 kernel launches, so they are directly comparable:

| category | FSS (arm B) | all-global | delta |
|---|---|---|---|
| gemm | 8.173 s | 8.106 s | -0.8% |
| norm | 1.598 s | 1.613 s | +0.9% |
| other | 1.419 s | 1.426 s | +0.5% |
| elementwise | 0.259 s | 0.257 s | -0.8% |
| attention | 1.849 s | 1.990 s | **+7.6%** |
| comm | 1.424 s | 3.255 s | **+129%** |

The gate's own per-run numbers say the same: compute 13212 -> 13300 ms
(**+0.67%**, inside noise) while nccl goes 1424 -> 3255 ms and exposed 135 ->
392 ms. **All-global cannot change communication** -- same parameters, same FSDP
sharding, same gradient volume -- so the comm term is the node draw, and the
gate's exclude list had grown by eight nodes between the two submissions. Both
runs passed the gate (1.0% and 2.9% exposed/compute against an 8% threshold),
which is the point: the gate catches exposed stalls, not a network that is
uniformly slower but still overlapped.

**Two things follow.**

**The alternating local/global pattern is not earning its place on compute.** On
FLOPs, 21 of 32 layers at window +-64 instead of full 512 should save about half
the attention cost. Measured, it saves 7.6% of attention, which is ~20 ms/step,
under 1% of a step. At sequence length 512 and head_dim 64 these FA3 kernels are
not FLOP-bound, so the window removes work the kernel was not spending time on.
The kernel counts confirm the structure rather than a measurement artifact: FSS
splits attention across two kernel variants (601+315 backward, 588+308 forward,
windowed and not) where all-global uses one (916 backward, 896 forward), the
same total number of attention calls.

**So all-global belongs in a fixed-step comparison, and the arm was relaunched
as one.** The matching rule's test is whether compute per step changes; it
changes by 0.67%, inside noise. Under wall-clock matching the arm was being
charged ~2.5-3% of step time for a network draw, which is exactly the
node-speed-into-downstream leakage this series measured at r=0.89 and wrote the
fixed-step rule to avoid. Relaunched at `max_steps: 10500` against arm B's
10500, with `max_wallclock_hours: 6.5` as a crash guard only. The first
submission (job 1766458) was cancelled an hour in.

Method note for later arms: *predicted* compute neutrality is not the test, and
neither is measured step time on one node draw. The trace decides, and the
category breakdown separates "this arm costs more" from "this node is slower" in
a way total step time cannot.

## newPISCES364, and why PBC_SUPERVISED had never run

**newPISCES364 is a test split of PBC's secondary-structure task**, not a contact
set: `SET=newPISCES364` on 364 of the 11205 sequences in
`secondary_structure.fasta`, alongside casp12 (20), casp14 (17), casp13 (12),
train (9712) and val (1080) -- and **no `SET=test` at all**, unlike every other
PBC dataset (`scl` ships train/val/test). Provenance per the dataset README:
FLIP-sampled following the ProtT5 paper, cited to Klausen 2019 (NetSurfP-2.0).
One sequence overlapping CASP14 was kept here and dropped from CASP14.

Biotrainer needs no change to report it. `get_split_lists` ends with
`case _: # Treat all other sets as testing sets` and `testing_ids` is a dict, so
every non-train/val/pred set name becomes its own named test set. The `splits`
field on the dataset registry is for datasets spread across separate *files* and
is unrelated. Results on the new base (arm B, dev mode, 3-state accuracy):

| test set | accuracy | balanced accuracy |
|---|---|---|
| **newPISCES364** | **0.8032** | 0.7969 |
| casp13 | 0.8342 | 0.8385 |
| casp12 | 0.7573 | 0.7573 |
| casp14 | 0.7457 | 0.7494 |

The whole supervised menu arrives in the same run, which is how `scl` finally has
a number too (0.589 balanced accuracy, validation).

**Why it had never run, which is the part worth keeping.** Three environmental
faults in a row, none of them in our code and none visible from the framework
list:

1. **Booster compute nodes have no outbound network.** The first attempt died on
   `Errno 101 Network is unreachable` fetching the dataset bundle from nextcloud.
   `force_download` on a compute node can only fail; any new PBC dataset has to
   be pulled from a login node first. The 26 MB archive now sits in
   `eval/data/PBC_SUPERVISED/`.
2. **The venv violates biotrainer's own dependency pin.** biotrainer requires
   `ruamel.yaml>=0.17.40,<0.18.0`; the venv had 0.19.1, which deleted the
   module-level `yaml.dump`/`yaml.load` API that biotrainer calls in at least two
   places. Every supervised task therefore crashed while writing its `out.yml`,
   which is why this framework had never completed a single task in this series.
   Fixed by installing 0.17.40 into `eval/pydeps` with `--target`, so it shadows
   the venv only under the eval `PYTHONPATH`; nanoplm does not import ruamel at
   all, so training is untouched, and nothing here touches metric computation.
3. **A crashed supervised run poisons the next one.** It leaves a 30 GB
   embeddings HDF5 (next run: `Unable to create dataset (name already exists)`)
   and a **0-byte `out.yml`**, which biotrainer's load-existing-output path will
   happily find. Wipe the output dir before any retry.

**Cost, before this becomes a standing column.** 34 min per arm for all 7 tasks,
of which ~13 min is embedding, plus **~30 GB of per-residue embeddings** written
to the output dir per arm. Fine for a handful of arms; not something to attach
to all 67 worklist rows without pruning the embedding files between runs.

## All-global attention: worse or tied on everything, and not adopted

`attn_layer_pattern: F` (32 full-attention layers) against the FSS base (11
full, 21 sliding at +-64), both off the modernized base at 2e-2, both
fixed-step to 10500, both scored on the full battery in dev mode:

| metric | FSS | all-global | delta |
|---|---|---|---|
| eval loss @10500 | 2.1748 | 2.1772 | +0.0024 (worse) |
| PGYM total SCC | 0.3694 | 0.3552 | -0.0142 |
| zero-shot contact, sel long P@L | 0.4283 | 0.4190 | -0.0093 |
| **supervised contact, sel long P@L** | **0.4560** | **0.4557** | **-0.0003** |
| supervised contact, casp14 long P@L | 0.2153 | 0.2082 | -0.0071 |
| **newPISCES364 accuracy** | **0.8032** | **0.7948** | **-0.0084** |
| casp13 / casp12 / casp14 accuracy | 0.8342 / 0.7573 / 0.7457 | 0.8285 / 0.7162 / 0.7230 | -0.0057 / -0.0411 / -0.0227 |
| scl balanced accuracy | 0.5655 | 0.5301 | -0.0353 |

**On the deciding metric it is a tie**: 0.0003, a seventh of `2*sigma_seed`. The
pre-registered rule therefore returns "no difference", and on that alone
all-global would be a wash. Everything else breaks the tie in one direction:
eight of the remaining metrics are negative and none is positive, which under a
sign test is p ~ 0.004 if each were a coin flip, and several are far outside
noise on their own (scl -0.035, casp12 -0.041, PGYM -0.014 at roughly 7 sigma).

**Not adopted.** It is worse or tied on every measure and it costs more: 21287 s
against 20409 s for the same 10500 steps. Only ~0.67% of that is real compute,
the rest being the node draw
([findings.md](findings.md#the-sliding-window-buys-1-of-step-time-and-a-36-step-time-gap-was-the-network)),
but there is no term on the other side of the ledger to pay for it.

**What this says together with the trace result.** The sliding window saves under
1% of step time, so the alternating pattern was never earning its place on
compute; and removing it does not help quality either. The honest reading is that
at 512 tokens and hidden 1024 the attention pattern barely matters in either
direction, with the alternating default very slightly ahead. Two consequences
worth carrying: the window is not a meaningful lever at this scale, so tuning it
further is not worth runs; and if sequence length grows in a later series both
halves of this conclusion have to be re-measured, because the window's FLOP
saving scales with L while its wall-clock saving here did not.
