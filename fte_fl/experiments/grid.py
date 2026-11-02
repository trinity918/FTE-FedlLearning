"""Run a grid of simulations (runs x seeds) described by a YAML spec.

    python -m fte_fl.experiments.grid fte_fl/experiments/grids/<spec>.yaml \
        [--jobs 3] [--only SUBSTRING] [--job-mb 700] [--min-available-mb 1024] \
        [--settle-s 90] [--poll-s 30] [--sample-s 60] [--dry-run]

Spec keys:
  base_config  YAML config every run starts from
  output_dir   runs go to <output_dir>/<run name>/seed<seed>
  seeds        list of project seeds
  common       {dotted.key: value} overrides for every run
  runs         {run name: {dotted.key: value}} per-run overrides
Each job is `python -m fte_fl.experiments.run` in its own process.  A job
whose summary.json says "completed" is skipped; one whose summary.json says
"failed" (e.g. the model diverged) is a recorded result and is not rerun (it
counts as failed; move the directory aside by hand to retry); a job whose
directory holds a config.json is resumed (`--resume`, which checks the config
hash); a non-empty directory without config.json (the run failed before
writing it, e.g. out of memory while loading data) is renamed to
<run_dir>.failed-<timestamp> and the job starts fresh; others start fresh.
Memory: a FEMNIST CnnSmall process peaks at ~620 MB RSS (1.5 GB before the
loader converted each JSON file to arrays as it is read).  Launches are serialised; before
each one the runner reads the available memory (MemAvailable from
/proc/meminfo and, under WSL, the Windows host's Available MBytes via
powershell.exe: the VM's own figure does not show host pressure) and
launches only if min(available) - --job-mb >= --min-available-mb, then waits
--settle-s so the next check sees the new job.  If memory is short while
jobs of this grid run, it waits for one to finish; if it is short with none
running, or cannot be read at all, it launches nothing more (running jobs
finish) and the remaining jobs are reported as not_started_low_memory /
not_started_memory_unknown.  Memory changes caused by other programs after
a launch are not controlled.  Launches, memory readings and every job's
exit code and elapsed time are appended to <output_dir>/_grid_log.jsonl as
they happen, and a memory reading every --sample-s seconds to
<output_dir>/_memory_log.jsonl.  Exit status is 1 if any job failed or was
not started.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def expand(spec: dict) -> list[dict]:
    """Jobs as {name, seed, run_dir, overrides} in spec order."""
    for key in ("base_config", "output_dir", "seeds", "runs"):
        if key not in spec:
            raise ValueError(f"grid spec is missing {key!r}")
    unknown = set(spec) - {"base_config", "output_dir", "seeds", "common", "runs", "description"}
    if unknown:
        raise ValueError(f"unknown grid spec key(s): {sorted(unknown)}")
    common = spec.get("common") or {}
    jobs = []
    for name, overrides in spec["runs"].items():
        for seed in spec["seeds"]:
            merged = {**common, **(overrides or {}), "project.seed": seed}
            jobs.append({"name": name, "seed": seed,
                         "run_dir": str(Path(spec["output_dir"]) / name / f"seed{seed}"),
                         "overrides": merged})
    return jobs


def command(spec: dict, job: dict, resume: bool) -> list[str]:
    cmd = [sys.executable, "-m", "fte_fl.experiments.run", "--config", spec["base_config"],
           "--run-dir", job["run_dir"]]
    for key, value in job["overrides"].items():
        cmd += ["--set", f"{key}={json.dumps(value)}"]   # JSON is valid YAML
    return cmd + (["--resume"] if resume else [])


def state(run_dir: Path) -> str:
    summary = run_dir / "summary.json"
    status = json.loads(summary.read_text())["status"] if summary.exists() else None
    if status == "completed":
        return "completed"
    if (run_dir / "config.json").exists():
        return "failed" if status == "failed" else "resume"
    if run_dir.exists() and any(run_dir.iterdir()):
        return "failed_early"     # failed before config.json was written
    return "new"


def set_aside(run_dir: Path) -> Path:
    """Rename an early-failure directory (kept as evidence) so a fresh run can start."""
    target = run_dir.with_name(f"{run_dir.name}.failed-{time.strftime('%Y%m%dT%H%M%S')}")
    run_dir.rename(target)
    return target


HOST_QUERY = ("(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes")


def under_wsl() -> bool:
    return Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def windows_available_mb() -> float:
    """Available physical memory of this Windows machine (GlobalMemoryStatusEx)."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    *[(n, ctypes.c_uint64) for n in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]]

    status = MemoryStatusEx(dwLength=ctypes.sizeof(MemoryStatusEx))
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / (1024.0 * 1024.0)


def memory_mb() -> dict:
    """Available memory in MiB: {"guest": MemAvailable from /proc/meminfo (None
    on native Windows), "host": the Windows host's available memory on native
    Windows and under WSL (the VM's own figure does not show host pressure),
    else None}.  Raises RuntimeError when a figure cannot be read."""
    if sys.platform == "win32":
        return {"guest": None, "host": windows_available_mb()}
    guest = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            guest = int(line.split()[1]) / 1024
    if guest is None:
        raise RuntimeError("/proc/meminfo has no MemAvailable line")
    host = None
    if under_wsl():
        try:
            res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", HOST_QUERY],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"cannot query Windows host memory: {exc}") from exc
        if res.returncode != 0 or not res.stdout.strip().isdigit():
            raise RuntimeError(f"cannot query Windows host memory: exit {res.returncode}, "
                               f"output {res.stdout.strip()!r} {res.stderr.strip()[:200]!r}")
        host = float(res.stdout.strip())
    return {"guest": guest, "host": host}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run a grid of simulations")
    parser.add_argument("spec")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only", default=None, help="run only jobs whose name contains this")
    parser.add_argument("--job-mb", type=float, default=700,
                        help="memory one job needs (FEMNIST CnnSmall: 622 MB peak RSS, measured)")
    parser.add_argument("--min-available-mb", type=float, default=1024,
                        help="memory that must remain available after a launch")
    parser.add_argument("--settle-s", type=float, default=90,
                        help="wait after each launch so the next check sees the job's memory")
    parser.add_argument("--poll-s", type=float, default=30)
    parser.add_argument("--sample-s", type=float, default=60,
                        help="append a memory reading to _memory_log.jsonl this often (0: never)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(Path(args.spec).read_text())
    jobs = [j for j in expand(spec) if args.only is None or args.only in j["name"]]
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = out / "_grid_log.jsonl"
    stop = []                         # first reason memory was insufficient or unknown: launch nothing more
    launch_lock, log_lock, count_lock = threading.Lock(), threading.Lock(), threading.Lock()
    running = [0]

    def write_log(entry):
        entry = {**entry, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with log_lock, open(log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def admit(job) -> str | None:
        """Wait until memory allows one more job; None to launch, else the reason not to."""
        while True:
            if stop:
                return stop[0]
            try:
                mem = memory_mb()
            except RuntimeError as exc:
                stop.append("not_started_memory_unknown")
                write_log({"name": job["name"], "seed": job["seed"], "event": "memory_unknown",
                           "error": str(exc)})
                return stop[0]
            avail = min(v for v in mem.values() if v is not None)
            if avail - args.job_mb >= args.min_available_mb:
                write_log({"name": job["name"], "seed": job["seed"], "event": "launch",
                           "memory_mb": mem, "running": running[0]})
                return None
            if running[0] == 0:       # nothing of ours will finish and free memory
                stop.append("not_started_low_memory")
                write_log({"name": job["name"], "seed": job["seed"], "event": "low_memory",
                           "memory_mb": mem, "running": 0})
                return stop[0]
            time.sleep(args.poll_s)   # a running job's exit may free enough

    def work(job):
        st = state(Path(job["run_dir"]))
        if st == "completed":
            return job, "skipped", 0, 0.0
        if st == "failed":
            return job, "failed_recorded", 1, 0.0
        cmd = command(spec, job, resume=st == "resume")
        if args.dry_run:
            print(" ".join(cmd))
            return job, "dry-run", 0, 0.0
        with launch_lock:
            reason = admit(job)
            if reason is not None:
                return job, reason, 1, 0.0
            if st == "failed_early":
                set_aside(Path(job["run_dir"]))
            Path(job["run_dir"]).parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            fh = open(f"{job['run_dir']}.out", "a")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            with count_lock:
                running[0] += 1
            time.sleep(args.settle_s)
        try:
            rc = proc.wait()
        finally:
            fh.close()
            with count_lock:
                running[0] -= 1
        return job, st, rc, time.perf_counter() - t0

    def logged(job):
        job, st, rc, elapsed = work(job)
        entry = {"name": job["name"], "seed": job["seed"], "start_state": st, "exit": rc,
                 "elapsed_s": round(elapsed, 1)}
        if st in ("skipped", "failed_recorded", "dry-run"):
            print(json.dumps(entry), flush=True)
        else:
            write_log(entry)          # written as soon as the job ends
        return rc

    done = threading.Event()

    def sample():
        """Memory trace while the grid runs (evidence if the grid is killed from outside)."""
        while not done.is_set():
            try:
                entry = {"memory_mb": memory_mb()}
            except RuntimeError as exc:
                entry = {"error": str(exc)}
            entry.update(running=running[0], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(out / "_memory_log.jsonl", "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            done.wait(args.sample_s)

    sampler = threading.Thread(target=sample, daemon=True)
    if args.sample_s > 0 and not args.dry_run:
        sampler.start()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failed = sum(rc != 0 for rc in pool.map(logged, jobs))
    done.set()
    if sampler.is_alive():
        sampler.join()
    print(f"{len(jobs)} jobs, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
