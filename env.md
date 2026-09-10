# Environment

## Pinned versions

```
# sep07 ablation env, built 2026-09-07T23:41:03+02:00
nanoplm_sha: 6cebfbdf31d7351a925302bf7ff6d2fbe2de8fb6
nanoplm_branch: abl/sep07-base
venv: /e/fscratch/profound/naeimitabiei1/sep07_abl/env/venv
python: Python 3.12.13
torch:                 2.12.1+cu130
transformer_engine:    2.16.0+4220403
flash_attn_3:          3.0.0+20260520.cu130torch2120cxx11abitrue.417891
dion:                  0.1.0
quack-kernels:         0.5.0
nvidia-cutlass-dsl:    4.5.3
sonic-moe:             0.1.2.post1

# TE fused RoPE was evaluated and reverted (neutral; see issue #148).
# The implementation lives on branch perf/te-fused-rope, not here.
```

The SHA is **not final**. It moves once the remaining prerequisites land
(see `plan.md`). When it does, update `PINS.txt`,
`pkgs/nanoplm/.FROZEN_SHA` and the frozen tree together.

## Layout

Everything a run touches lives on fscratch. Nothing on /e/project1, which is
near its inode limit.

```
/e/fscratch/profound/naeimitabiei1/sep07_abl/
  configs/       one yaml per arm
  slurm/         sbatch_run.sh
  logs/          job stdout/stderr + nvidia-smi samples
  checkpoints/   per-run, includes profiler_traces/
  cache/         inductor, triton, wandb, quack, per-job TMPDIR
  run/           cwd for jobs. Deliberately empty: a dion/ directory in the
                 nanoPLM root shadows the installed dion package
  data/          the pretraining corpus
  env/venv       the venv, staged off project fs
  pkgs/nanoplm   frozen source, .FROZEN_SHA pins it
  design/        pointer to this repo
```

## Launcher

`slurm/sbatch_run.sh`. Things in it that are load-bearing:

- `#SBATCH --nodes=4`. Without it sbatch silently allocates one node, the
  pipeline raises grad_accum to compensate, and the run reports a much better
  MFU that is not comparable to anything.
- `srun --overlap` for the nvidia-smi sampler. A plain srun step holds the
  node's resources and starves the training step.
- `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`. About +2.3 MFU points, paid in
  compile time, which is outside the wall-clock budget.
- per-job TMPDIR on fscratch, and every cache redirected there.

## Evaluation

`nanoplm eval` runs biotrainer's autoeval. The released biotrainer 2.0.0 cannot
run the contact framework at all (`TypeError: Got unsupported ScalarType
BFloat16`), so the venv carries **PR #192** (`peymanvahidi/biotrainer`,
`fix/v2-migration-findings`) pinned at `ed33f6a`, editable from
`sep07_abl/pkgs/biotrainer-pr192`, installed `--no-deps` so it never re-resolves
the quack/cutlass pins the MoE kernels need. nanoPLM side, commit `7fa8d16`
carries two fixes that change the numbers: the token budget was capping a
configured batch, and the categorical Jacobian had no context-length guard, so
it would have scored proteins at lengths the model never trained on.

```
sep07_abl/eval/
  eval-*.yaml    one config per evaluated checkpoint
  data/          benchmark datasets (biotrainer custom_storage_path)
  out/           autoeval reports and per-framework caches
  logs/
```

Settings that matter: `zero_shot_batch_size: 1024`, `zero_shot_method:
masked_marginals`, `torch_compile: true`.

### Making it 5x cheaper

The first working run took 31.5 min per checkpoint in development mode. The full
benchmark is much bigger (PGYM 86 -> 217 assays, contacts 147 -> 1621 proteins),
so at that speed it would have been ~5.5 h per checkpoint and ~280 GPU-hours for
52 arms, about three training runs. Three things were wrong:

1. **The configured batch size was not the batch size.** A 65,536-token budget
   capped a 1024-token scoring window at 64 rows, so a requested 256 silently
   ran as 64. Budget raised to 262,144.
2. **256 underfeeds the GPU anyway.** A typical contact protein is ~300
   residues, so 256 rows is only 76k tokens per forward. Requesting 1024 (the
   budget then caps the effective batch at 865) is 20% faster; above that it
   saturates.
3. **Eval ran the model in eager mode.** A profile showed only **27.7% of GPU
   time in matmuls**, despite GEMMs of 262144 x 1024 x 2688 that should run near
   peak. The rest was unfused elementwise work: 20.1% `mul` (RoPE and gating),
   13.3% `copy_`, 13.0% `add`, 8.6% `gelu`, 9.9% `neg`+`cat` for rotate_half.
   Training runs this model compiled; eval did not, and paid for it on every one
   of the 20*L forwards the Jacobian needs per protein.

`torch.compile(dynamic=True)` took the forward from **185k to 375k tok/s**
(148 -> 300 TFLOP/s), in line with training. Dynamic shapes are required, not
optional: every protein is a different length. Three unseen shapes measured
after a 12 s warmup all ran at full speed, so shape variation costs nothing
after the first few seconds. Validated against eager on the same checkpoint and
subset, all 54 report metrics: PGYM total scc differed by 0.0001, contact P@L by
<=0.0012, largest deviation anywhere 0.0025 on a metric whose bootstrap CI is
+-0.08. End to end 1891 s -> 1091 s. Net ~2.7 h per checkpoint on the full
benchmark, ~143 GPU-hours for 52 arms.

### Running it across the series

`slurm/sbatch_eval.sh` is a job array: one node per array element, one eval
process per GPU, each task taking every Nth line of the worklist. Load-bearing
details:

- Slurm 25.05.9 binds a single GPU per task, which the training launcher has to
  work around by unsetting `CUDA_VISIBLE_DEVICES`. Here it is exactly what is
  wanted, so it is left alone.
- The inductor cache is shared across tasks on purpose: same architecture, so
  the first compile serves all 52 arms.
- Materialise the benchmark task lists once before submitting
  (`eval/tools/warm_datasets.py`), or 52 tasks race to preprocess the same
  `dataset_dir`.
- Eval loss comes from each run's Slurm log, at the nearest eval step at or
  before the checkpoint step, plus a common-step column.

## FlashAttention-3 A/B builds

Builds live in `fa_gh200/abx/{stock,fork,forkflag}`, built by
`abx/build_min.sbatch`; A/B is `abx/ab.sbatch`, analysis `abx/analyse_ab.py`,
kernel parity `abx/parity.py`.

- Only the corner nanoPLM calls is compiled (varlen, bf16, headdim 64, MHA,
  sliding-window, fwd+bwd): 4 minutes, against ~8 hours for the full
  451-instantiation matrix. `LOCAL` must stay enabled.
- Two build traps: the fork calls `git submodule update` unconditionally in
  `setup.py` where the merge-base guards it, so compute nodes need
  `module load git`; and `csrc/cutlass` must be populated beforehand, because
  compute nodes have no outbound network.
- Each arm is selected with `PYTHONPATH=<tree>/hopper`, which overrides the
  installed `flash_attn_3`. **Verify the three `_C.abi3.so` md5s differ**
  (`3008de034885` / `d880c883f129` / `84db0afd17c4`): a PYTHONPATH that fails to
  override would make every arm identical and the A/B a silent no-op.
