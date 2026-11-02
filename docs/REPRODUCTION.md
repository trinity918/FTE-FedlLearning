# Reproducing the results

Commands run from the repository root on native Windows, in Git Bash, with
`.venv` active. Setup is in `README.md` (Environment, Datasets). Runs never
overwrite: a finished job is skipped, a partial one is resumed if its
configuration hash matches, and a run that recorded a failure stays as a result.

## Before a long run

- Memory: launches need at least 3.5 GB available. `scripts/run_queue.sh` passes
  `--min-available-mb 2800 --job-mb 700` and runs 3 jobs; set `JOBS=2 JOB_MB=1800`
  for Speech Commands, whose jobs peak near 1.9 GB.
- Run one workload at a time, and keep the laptop on mains power.
- Never run two grids over the same runs: `grid.py` has no lock, and two writers
  produce a run marked completed with extra metric lines. `analyze` refuses these.
  Before starting a queue, confirm no earlier one is alive. Stopping a parent
  shell does not necessarily stop its `run_queue.sh`.
- Windows' Best power efficiency mode can make background runs about 2.5x slower.
- Take timing benchmarks with nothing else running.
- Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The
  queue script does this itself.

## FEMNIST

```bash
bash scripts/run_queue.sh \
    "fte_fl/experiments/grids/femnist_main_lr0.1.yaml@alie__" \
    "fte_fl/experiments/grids/femnist_main_lr0.1.yaml@backdoor__" \
    "fte_fl/experiments/grids/femnist_main_lr0.1.yaml@labelflip__" \
    fte_fl/experiments/grids/femnist_stratified_control.yaml \
    fte_fl/experiments/grids/femnist_platform_attack_check.yaml \
    "fte_fl/experiments/grids/femnist_main_lr0.1.yaml@gaussian__" \
    "fte_fl/experiments/grids/femnist_main_lr0.1.yaml@zero__"
```

Fourteen runs in `femnist_stratified_control` and `platform_check` were invalidated
by two queues writing to the same directories (`results/WINDOWS_PROGRESS.md`, 19:53).
They are kept but excluded. Their replacements came from
`femnist_stratified_control_rerun.yaml` and `femnist_platform_attack_check_rerun.yaml`.
A clean reproduction needs only the two original grids above.

The first 45 runs in `results/runs/m6/femnist_main_lr0.1` ran under WSL2 and the
rest on Windows. Results are not bitwise identical across platforms; see
Milestone 6 in `results/PAPER_NOTES.md`.

## Speech Commands

Build the feature cache first (README, Datasets). LR selection, then the grid:

```bash
JOBS=2 JOB_MB=1800 bash scripts/run_queue.sh fte_fl/experiments/grids/speech_pilot_v2.yaml
JOBS=2 JOB_MB=1800 bash scripts/run_queue.sh fte_fl/experiments/grids/speech_main_bcres_seed0.yaml
JOBS=2 JOB_MB=1800 bash scripts/run_queue.sh fte_fl/experiments/grids/speech_main_bcres_seed1.yaml
JOBS=2 JOB_MB=1800 bash scripts/run_queue.sh fte_fl/experiments/grids/speech_main_bcres_seed2.yaml
python -m fte_fl.eval.full_eval results/runs/m6_windows/speech_main_bcres
```

A stage reports exit 1 when any job failed, and the queue moves on to the next
stage. On Speech Commands that is expected
for some attacks: BatchNorm running statistics are aggregated, and an attacked
aggregate can leave a negative `running_var`. `fte_fl/fl/state.py` deliberately
fails on this instead of clamping, so check `summary.json` and report such runs
as diverged.

## Protocol benchmarks

Each benchmark reads a run's own configuration, saved as YAML from `raw` in its
`config.json`, so the updates match that run. All protocols then aggregate the same
update matrix, and EtF Krum dominates the runtime (about 6 minutes per FEMNIST
round). Run with nothing else on the machine:

```bash
B=results/runs/m6_windows/protocol_bench
python -m fte_fl.experiments.protocol_bench --config $B/scaling__multikrum_seed0_config.yaml \
    --out $B/femnist --checkpoint results/runs/m6/femnist_main_lr0.1/scaling__multikrum/seed0/checkpoint.pt \
    --rounds 101 102 103 --budget 4
python -m fte_fl.experiments.protocol_bench --config $B/speech_scaling__multikrum_seed0_config.yaml \
    --out $B/speech --checkpoint results/runs/m6_windows/speech_main_bcres/scaling__multikrum/seed0/checkpoint.pt \
    --rounds 41 42 43 --budget 2
```

The global model is not updated between benchmark rounds, so each round is an
independent sample of participants at the same model state.

## Analysis and paper artifacts

Give each analysis its own `--out` directory, starting with `_`, so earlier analyses
are kept. Analysis refuses incomplete groups and lists them in `problems.txt`.

```bash
python -m fte_fl.eval.analyze results/runs/m6/femnist_main_lr0.1 \
    --out results/runs/m6/femnist_main_lr0.1/_analysis_windows_final
python -m fte_fl.eval.paper_artifacts \
    --femnist results/runs/m6/femnist_main_lr0.1 \
    --control results/runs/m6_windows/femnist_stratified_control               results/runs/m6_windows/femnist_stratified_control_rerun \
    --speech bcres=results/runs/m6_windows/speech_main_bcres \
    --bench results/runs/m6_windows/protocol_bench \
    --out paper
python paper/check_tex.py
python -m fte_fl.eval.platform_check     results/runs/m6/femnist_main_lr0.1/scaling__multikrum/seed0     results/runs/m6_windows/platform_check_rerun/scaling__multikrum/seed0
```

Compiling the paper is covered in `paper/README.md`.

## Tests

Do not run these while experiments are running.

```bash
OMP_NUM_THREADS=2 python -m pytest tests -q
```
