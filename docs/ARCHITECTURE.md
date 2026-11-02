# Architecture

All parties run in one process. There is no network: traffic is counted from
serialized sizes.

## One training round

`Simulation` in `fte_fl/fl/simulator.py`:

1. `sample_participants` picks clients. Honest runs sample uniformly; attacked runs
   are stratified, with a fixed number of malicious clients per round.
2. `local_train` trains each client from the global model.
3. Attacks in `fte_fl/fl/attacks.py` poison the data before training or transform
   the update afterwards.
4. `aggregate` in `fte_fl/fl/aggregation.py` applies the rule, either in plaintext
   or through a protocol in `fte_fl/defenses/protocols.py`.
5. `set_vector` in `fte_fl/fl/state.py` installs the new global model. It fails on a
   negative BatchNorm `running_var` and does not clamp it.
6. The round's metrics go to `metrics.jsonl`, and a checkpoint is saved for resuming.

## Run directories

| File | Content |
|---|---|
| `config.json` | raw and parsed config, config hash, environment |
| `manifest.json` | client ids, sample counts, malicious clients, poisoning |
| `metrics.jsonl` | one line per round |
| `checkpoint.pt` | global vector after the last completed round |
| `summary.json` | `status`: `running`, `completed` or `failed` |

The raw config, including data paths, is part of the hash, and a resume validates
it. Do not edit `fte_fl/configs/default.yaml`: every existing run would stop
resuming. `native_path` in `fte_fl/fl/data.py` maps `/mnt/c` paths at load time.

## Execution and analysis

- `fte_fl/experiments/grid.py` expands a grid YAML and launches runs under a RAM
  gate. It has no lock, so never run two grids over the same runs.
- `scripts/run_queue.sh` runs stages in order and retries only memory-gated stages.
- `fte_fl/experiments/protocol_bench.py` runs every protocol on identical updates.
- `fte_fl/eval/full_eval.py` scores Speech checkpoints on the full test split and
  refuses a checkpoint that does not replay its recorded subset accuracy.
- `fte_fl/eval/analyze.py` validates runs: exact round counts, finite metrics,
  checkpoints. It refuses a group that fails.
- `fte_fl/eval/platform_check.py` compares a run with its replay on another platform.
- `fte_fl/eval/paper_artifacts.py` builds the paper's tables and figures.

## Further reading

| Topic | Location |
|---|---|
| Windows setup, datasets, models, attacks, rules | `README.md` |
| Execution and reproduction | `docs/REPRODUCTION.md` |
| Limitations | `paper/sections/limitations.tex`, `results/PAPER_NOTES.md` |
| Compiling the paper | `paper/README.md` |
| What happened during the Windows runs | `results/WINDOWS_PROGRESS.md` |
