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
(see `ablation-ladder.md`). When it does, update `PINS.txt`,
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

`nanoplm eval` runs biotrainer's autoeval. The released biotrainer 2.0.0
cannot run the contact framework, so the venv carries **PR #192**
(`peymanvahidi/biotrainer`, `fix/v2-migration-findings`) pinned at `ed33f6a`,
editable from `sep07_abl/pkgs/biotrainer-pr192`, installed `--no-deps` so it
never re-resolves the quack/cutlass pins the MoE kernels need.

```
sep07_abl/eval/
  eval-*.yaml    one config per evaluated checkpoint
  data/          benchmark datasets (biotrainer custom_storage_path)
  out/           autoeval reports and per-framework caches
  logs/
```

Runs on one login-node GH200; 31.5 min for PGYM + PBC contact in development
mode. See `results-eval.md`.
