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

## Why NCCL time ballooned in the TE RoPE run

It did not. NCCL kernel *duration* includes the time a rank sits inside the
collective waiting for its peers. The fused kernel made the compute before the
all-gather faster, so ranks arrived earlier and waited longer, and the recorded
NCCL time grew from 628 ms to 1704 ms while step time did not move at all.

The collective was the critical path the whole time. Total NCCL time is not a
cost; only exposed (non-overlapped) comms is.

## The lambdas are bf16 in the forward, and that is fine

`resid_lambdas` and `x0_lambdas` are fp32 parameters. Under FSDP2 with
`param_dtype=bf16` they are bf16 in the forward with fp32 masters in the
optimizer, which is the ordinary mixed-precision path every weight takes. It is
NOT the `cast_forward_inputs` problem that broke the RoPE angles, because those
were non-parameter tensors passed as forward args at large magnitudes.

Values near 1.0 have ~0.4% spacing in bf16 and updates accumulate in the fp32
master. Verified under a real 1-rank FSDP2 with `MixedPrecisionPolicy`.

## nanoplm deviates from dion's optimizer defaults on four knobs

`adjust_lr` (rms_norm vs spectral_norm), `cautious_wd` (true vs false),
`nesterov` (true vs false), `epsilon` (1e-7 vs 1e-8). None is documented as
deliberate. The series now runs dion's defaults, and the last two are arms in
Tier 1 Wave 2 so the deviation gets tested rather than inherited.

## MoE always has a shared expert, and it changes the arithmetic

`MoELayer` builds one `ModernBertSwiGLUMLP(config)` that processes every token
(`moe.py:468`), at the same `intermediate_size` as a routed expert. There is no
knob to turn it off.

So active experts per token is `top_k + 1`, not `top_k`, and:

- active MLP width = `(top_k + 1) x intermediate_size`
- sparsity = `(moe_num_experts + 1) / (top_k + 1)`

This is why jul30's `arm11-moe12x` (48 routed, top_k 3) was reported as 49
experts and 12.25x. A matched-active grid that counts only routed experts is
wrong on both axes: it under-counts active width by 25% at top_k 3.

## Bigger global batch buys scale-out headroom, not comms savings

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

## TE's sync-free grouped GEMM: Blackwell-only in 2.15, Hopper from 2.16/2.17

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

## Nothing beats sonicmoe on Hopper, and the reason is not the GEMM

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
router-weight combine into Wo's epilogue. Every grouped-GEMM backend needs
`moe_scatter_dispatch` and `moe_gather_combine` around it, which at g8-S12 is a
materialized (458752, 1024) bf16 tensor, about 940 MB, touched roughly six
times across forward and backward. That traffic is the gap. Swapping the GEMM
cannot close it, which is why the field of candidates does not matter much.

The published record agrees: SonicMoE (ICLR 2026) beats ScatterMoE by 1.86x,
MoMoE, MegaBlocks, Megatron and DeepGEMM++ on H100, at intermediate size 256,
which is our 336 regime.

`torch._grouped_mm` is still worth knowing about. It is in our torch 2.12, runs
on GH200 sm90 bf16 with device-side offsets, handles ragged and empty groups,
and profiles as one CUTLASS sm9x grouped kernel per call with zero Memcpy DtoH.
It beats our JIT-built cutlass extension by 7 to 11% with no dependency and no
multi-rank build race.


## TE 2.18 on Hopper: built, measured, tied with torch._grouped_mm

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

## Slurm 25.05.9 broke three things at once (2026-09-08 maintenance)

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

## Node-speed variation is ~9% and it outweighs the effects we are measuring

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

## Two agent sessions will collide, and documentation does not stop it

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

## Classify an arm as compute-changing from the TRACE, never from intuition

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

## Gate every run on its trace at ~5 minutes, not at 6.5 hours

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

## Wall-clock matching contaminates downstream scores through step count

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
noise floor.** The real floor is roughly half what `11-results-eval-method.md` states.

Applying the same correction to Tier 1b flips three verdicts from tie to
separated, including cautious weight decay, which ran 733 fewer steps than the
base and moves from -0.0034 to +0.0058 once that is accounted for. **This is
not a result.** It is evidence that the raw downstream comparison within a tier
is confounded and cannot be trusted in either direction. Two cautions on the
correction itself: the threshold it produces rests on six points, and the
cautious arm stopped at 10303, which is 321 steps below the lowest replicate
(10624), so it carries the largest correction with the least support.

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

