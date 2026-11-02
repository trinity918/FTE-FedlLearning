#!/usr/bin/env bash
# Run a list of grid workloads one after another, at most --jobs 3 at a time.
#   [JOBS=2] [JOB_MB=1800] bash scripts/run_queue.sh "<grid yaml>[@<--only substring>]" ...
# Each launch still needs JOB_MB + 2800 MB available, so a third job waits while memory is short.
# grid.py stops launching if host memory is short with no job of its own
# running; a stage that ended that way is retried after a wait.  A stage with
# a failed run is not retried here (grid.py records it); the queue continues.
set -u
PY=.venv/Scripts/python.exe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for item in "$@"; do
    spec=${item%%@*}
    only=(); [ "$item" != "$spec" ] && only=(--only "${item#*@}")
    for attempt in 1 2 3 4 5 6; do
        output=$("$PY" -m fte_fl.experiments.grid "$spec" --jobs "${JOBS:-3}" "${only[@]}" --job-mb "${JOB_MB:-700}" \
                 --min-available-mb 2800 --settle-s 45 2>&1)
        status=$?
        echo "$output" | tail -1
        echo "[queue] $item attempt $attempt exit $status $(date +%H:%M:%S)"
        echo "$output" | grep -q 'not_started_low_memory' || break
        sleep 600
    done
done
