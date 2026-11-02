# Windows execution log

Times are local, 2026-11-01. Facts here come from run directories and grid logs.

| Time | Event |
|---|---|
| 01:05 | Takeover. Repo copied one way from WSL, 840/840 files SHA-256 verified. |
| 01:57 | Claude Code memory-pressure protection killed 5 Python jobs. My load, no results lost. |
| 08:20 | User authorized a restart. Limits: 2 jobs, one workload at a time, launch floor 3.5 GB. |
| 08:30 | zero__fedavg seeds 0-2 completed (resumed from rounds 8, 20, 14). Grid 48/120. |
| 08:31 | Speech learning-rate pilot started. |
| 08:45 | Stopped speech_lr_pilot myself after 6-7 rounds: BC-ResNet flat at chance (0.041) at lr 0.02 and 0.05, ~4 SGD steps per client per round. Partial runs kept, not completed, not used for any result. lr 0.1 and all AST runs never started. |
| 08:52 | speech_sanity (centralized, not a federated result): BC-ResNet val 0.236 after 400 steps, 45 ms/sample; LoRA-AST at lr 0.1 did not learn, 6 ms/sample. |
| 09:00 | LoRA-AST sweep, lr 0.001-0.03, 1500 steps: last-round val 0.050, 0.047, 0.045, 0.080. No full LoRA-AST grid: the random frozen backbone barely exceeds chance. |
| 09:01 | speech_pilot_v2 started: BC-ResNet lr 0.1 and 0.2, one LoRA-AST run; 11 clients, batch 8, 40 rounds. |
| 09:47 | Speech main grid seed 0 started. |
| 11:31 | All 3 seed 0 backdoor jobs exited at startup: my grid omitted `attack.speech_amplitude` and `attack.speech_freq_hz`, so no run directory exists. Added the trigger code's defaults (0.1, 1000 Hz) to both seed files. Other runs' configs did not change. |
| 11:48 | Seed 0 queue ended: 6 completed (honest x3, scaling Multi-Krum, scaling trimmed, signflip trimmed), 3 diverged (below), 3 backdoor never started. |
| 11:55 | Replayed round 1 of signflip Multi-Krum and scaling FedAvg in a scratch directory. Cause in both: negative BatchNorm `running_var` after aggregation, which is the fail-don't-clamp behaviour documented in `fte_fl/fl/state.py`. Scaling FedAvg: most variances negative. Signflip Multi-Krum: selected 1 of 2 flipped clients (client index 196), leaving 1 of 128 entries negative in `cnn_head.1`. Signflip FedAvg (round 11) was not replayed, so its cause is unverified. |
| 11:56 | Config left unchanged: the 3 diverged runs stay as measured failures. Machine idle, so the protocol benchmark validation ran before the next grid to get undisturbed timings. |
| 12:12 | `validation_femnist` hit my 900 s timeout after 14 of 20 records (one round, FEMNIST scaling Multi-Krum seed 0 checkpoint). The records are kept; EtF Krum and Multi-Krum take ~365 s each. |
| 12:15 | `validation_femnist_attested` measured the pairs still missing for one round: `fte` Bulyan and `fte_attested` Bulyan, Krum and Multi-Krum. Selection matched plaintext for all of them. `fte_attested` fedavg and trimmed mean have not been validated yet; the full benchmark covers them. |
| 12:16 | Three-round benchmarks started, FEMNIST (rounds 101-103, budget 4) then Speech (rounds 41-43, budget 2), nothing else running. |
| 13:05 | Found `scripts/run_queue.sh` hardcoded `--job-mb 700`, so `JOB_MB=1800` had no effect on Speech seed 0. Fixed. Effect on that grid: launches still met the 3.5 GB floor (lowest launch 4991 MB); lowest host reading 2891 MB, 1 of 104 samples below 3500 MB. |
| 14:03 | Benchmarks finished: FEMNIST 60/60 and Speech 60/60 records, exit 0. All 108 supported pairs matched plaintext selection; the other 12 records are EtF trimmed mean and Bulyan, unsupported. Speech EtF FedAvg and Multi-Krum differ from plaintext by up to 3.2e-5 against under 1e-7 for FTE; cause not established. |
| 14:04 | Laptop found on battery at 72% (status 1; on AC at 08:20). Power was not recorded during the benchmarks, so it cannot be ruled out as a cause of their timing spread. Runs have checkpoints and can resume. |
| 14:05 | Queue started: Speech seed 0 backdoors, full_eval, FEMNIST alie, backdoor, labelflip, control, platform check. |
| 16:15 | User approved 3 jobs if memory is available, up from the 2 promised at 08:20. Applied to FEMNIST stages via the `run_queue.sh` default; the launch floor is unchanged (3.5 GB). Speech seed 1 moved last and is optional; it stays at `JOBS=2`. |
| 17:07 | Speech seed 0 full_eval finished: 9 scored; the 3 diverged runs correctly refused. |
| 17:09 | FEMNIST ALIE at 3 jobs ran ~9 s/round against ~3.4 s earlier, with local training ~2.6x slower; each job had a full core. RAM was not the limit. Windows power overlay found at Best power efficiency (961cc777). |
| 17:10 | User asked to clear unnecessary processes: stopped windowless Edge, GooglePlayGamesServices and the AdBlock360 tray app (not its service). Available RAM 5577 to 6230 MB. Set the power overlay to Best performance (ded574b5). Restore with `powercfg /overlaysetactive 961cc777-2547-4f9d-8174-7d86181b8a7a`. |
| 18:46 | User demanded Speech now. Stopped the queue script at labelflip__bulyan; that grid and its 3 jobs kept running and were left to finish. Once they end: Speech seed 1 at 2 jobs plus remaining FEMNIST at 1 job, 3 in total. This runs two workloads at once, departing from the 08:20 one-workload limit; the 3.5 GB floor still applies to FEMNIST launches and 4.6 GB to Speech launches. |
| 18:48 | User asked for multiple seeds as long as RAM holds. Added `speech_main_bcres_seed2.yaml` (seed only differs) as a separate 1-job lane; each launch needs 4.6 GB available. This allows a fourth job, gated by RAM. |
| 19:33 | Seed 2 never launched: host RAM stayed below its 4.6 GB gate with 3 jobs running (3389 MB at this check), so its first lane logged `not_started_low_memory` on every attempt. No run directories or results exist for seed 2. Queued a second lane to start when FEMNIST ends. Seed 1 running ~80 s/round against ~50 s for seed 0. |
| 19:53 | **My error, found by `analyze`.** At 18:46 TaskStop killed only the outer shell; the inner `run_queue.sh` survived and ran control and platform stages at 3 jobs while my new lane ran the same stages at 1 job, in the same directories, from 18:48 to 19:44. Up to 6 jobs ran, not the 3 I reported, which likely explains the low RAM, seed 2 not launching, and Speech at ~80 s/round. Invalid: `femnist_stratified_control/noop__{fedavg,krum,multikrum,bulyan}/seed0-2` (105-177 metric lines) and `platform_check/{scaling__multikrum,signflip__trimmed}/seed0` (200, 109 lines). `summary.json` says completed, but they are excluded. Kept as they are, not deleted or reused. Valid: `noop__trimmed/seed0-2` and `platform_check/honest__fedavg`. Main grid and Speech audited: all finished runs have exact round counts. Stray queue exited 19:44. |
| 19:54 | Rerunning the 14 invalid runs into `femnist_stratified_control_rerun` and `platform_check_rerun`, 1 job, 3.5 GB floor. |
| 21:51 | All 14 reruns completed with exactly 100 metric lines. Control differences -0.013 to -0.005, all intervals include zero. Attacked replays against Linux, last-10 accuracy: scaling Multi-Krum -0.0006, sign flip trimmed -0.0021; largest single-round difference 0.0743. |
| 22:34 | Speech seed 1 queue ended: 12 jobs, 9 completed, 3 diverged (scaling FedAvg, signflip FedAvg, signflip Multi-Krum, as on seed 0). |
| 22:54 | FEMNIST main grid 120/120 completed, 0 failed, every run exactly 100 rounds; `analyze` reports 0 problems (`_analysis_windows_final`). Speech seed 2 started. |
| 22:58 | Zero attack on FEMNIST: Krum 0.054, Bulyan 0.094. Krum's aggregate was exactly zero in 98-99 of 100 rounds on all three seeds. |
| 23:19 | Speech seed 1 full_eval finished: 9 scored, 3 diverged runs refused (exit 1 is expected). 18 Speech scores exist. Trimmed mean backdoor ASR: seed 0 0.0004, seed 1 0.777. |
| 23:25 | All paper sections written and tables regenerated with seeds 0 and 1. `paper/check_tex.py` had broken regexes and had never worked as committed here; repaired, reports ok. Paper not compiled. |
| 01:52 | Speech seed 2 queue ended: 12 jobs, 8 completed, 4 diverged (seed 0/1's three plus scaling Multi-Krum at round 8). |
| 03:09 | Seed 2 full_eval finished, 8 scored, 4 refused. 77 min against my 40 min estimate. Speech grid: 26 scores, 10 diverged. Paper updated to three seeds. |
| 03:20 | Full test suite: 647 passed, 0 failed. Power overlay restored to Best power efficiency (961cc777). All experiments finished; nothing is running. |
