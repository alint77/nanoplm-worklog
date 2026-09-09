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
