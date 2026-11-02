"""Tests for the federated simulator (fte_fl.fl), on synthetic data unless noted.

Stage A: config validation, sampling, update-vector convention, data.
Stage B: honest FedAvg equals the sample-weighted aggregate; held-out
         evaluation; bitwise reproducibility; checkpoint/resume.
Stage C: every defense end to end; invalid budgets rejected.
Stage D: EtF / FTE reconstruct the plaintext aggregate within CKKS error.
Stage E: FTE + attestation, honest and tampered submissions.
Stage F: attack integration touches only malicious clients/samples.
"""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from fte_fl.attacks import triggers
from fte_fl.defenses.aggregators import RULES, combine
from fte_fl.defenses.protocols import attest_update
from fte_fl.fl import attacks as fl_attacks
from fte_fl.fl import simulator as sim_mod
from fte_fl.fl.aggregation import aggregate
from fte_fl.fl.config import AttackConfig, parse_sim_config
from fte_fl.fl.data import TRAINING, ClientData, build_federated_data, derive_seed
from fte_fl.fl.metrics import filtering_metrics
from fte_fl.fl.simulator import Simulation, local_train, run_simulation, sample_participants
from fte_fl.fl.state import copy_excluded, get_vector, set_vector, vector_names, vector_size

SMALL_CKKS = {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60],
              "global_scale": 2**40, "chunk_size": 4096}


def _raw(rounds=2, k=10, defense=None, attack=None, protocol="plaintext", num_clients=10, **exp):
    raw = {
        "project": {"seed": 7},
        "crypto": dict(SMALL_CKKS),
        "experiment": {"dataset": "synthetic", "model": "mlp", "rounds": rounds,
                       "clients_per_round": k, "local_epochs": 1, "batch_size": 8,
                       "optimizer": "sgd", "lr": 0.1, "protocol": protocol, **exp},
        "defense": defense or {"name": "fedavg"},
        "attack": attack or {"type": "none"},
        "datasets": {"synthetic": {"num_clients": num_clients, "num_classes": 4,
                                   "samples_per_client": 20, "test_samples": 200, "dim": 8,
                                   "separation": 2.0}},
    }
    return raw


def _metrics(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def _strip_times(rec):
    rec = dict(rec)
    for key in ("time_s", "peak_rss_mb"):
        rec.pop(key)
    return rec


# ─── Stage A: configuration ──────────────────────────────────────────────────

@pytest.mark.parametrize("defense, n, match", [
    ({"name": "krum", "f": 4}, 10, r"n > 2f \+ 2"),
    ({"name": "multi_krum", "f": 1, "m": 11}, 10, "1 <= m <= n"),
    ({"name": "trimmed_mean", "b": 5}, 10, "n - 2b >= 1"),
    ({"name": "bulyan", "f": 2}, 10, r"n >= 4f \+ 3"),
    ({"name": "bulyan", "f": 0}, 10, "f >= 1"),
    ({"name": "krum"}, 10, "requires f"),
    ({"name": "trimmed_mean"}, 10, "requires b"),
    ({"name": "fedavg", "f": 1}, 10, "takes no"),
    ({"name": "krum", "f": 1, "b": 1}, 10, "takes no"),
    ({"name": "multikrum", "f": 1}, 10, "unknown defense"),
    ({"name": "krum", "f": 1, "fedavg_weighting": "uniform"}, 10, "applies only to fedavg"),
    ({"name": "krum", "f": 1.5}, 10, "must be an int"),
])
def test_invalid_defense_configs_rejected(defense, n, match):
    with pytest.raises(ValueError, match=match):
        parse_sim_config(_raw(k=n, defense=defense))


def test_boundary_participant_counts_accepted():
    parse_sim_config(_raw(k=7, defense={"name": "bulyan", "f": 1}))          # 4f + 3
    parse_sim_config(_raw(k=5, defense={"name": "krum", "f": 1}))            # 2f + 3
    parse_sim_config(_raw(k=5, defense={"name": "trimmed_mean", "b": 2}))    # n - 2b = 1
    with pytest.raises(ValueError):
        parse_sim_config(_raw(k=6, defense={"name": "bulyan", "f": 1}))
    with pytest.raises(ValueError):
        parse_sim_config(_raw(k=4, defense={"name": "krum", "f": 1}))


@pytest.mark.parametrize("name, params", [("trimmed_mean", {"b": 1}), ("bulyan", {"f": 1})])
def test_etf_rejects_coordinate_wise_rules(name, params):
    with pytest.raises(ValueError, match="protocol 'etf' supports only"):
        parse_sim_config(_raw(defense={"name": name, **params}, protocol="etf"))


@pytest.mark.parametrize("change, match", [
    ({"experiment": {"model": "cnn_small"}}, "not paired"),
    ({"experiment": {"dataset": "mnist"}}, "unknown dataset"),
    ({"experiment": {"protocol": "he"}}, "unknown protocol"),
    ({"experiment": {"optimizer": "adam"}}, "only optimizer 'sgd'"),
    ({"experiment": {"clients_per_round": 11}}, "exceeds num_clients"),
    ({"experiment": {"lr": 0}}, "must be > 0"),
    ({"experiment": {"rounds": 0}}, ">= 1"),
    ({"experiment": {"epochs": 1}}, "unknown key"),
    ({"defense": {"name": "fedavg", "beta": 1}}, "unknown key"),
])
def test_invalid_experiment_configs_rejected(change, match):
    raw = _raw()
    for section, values in change.items():
        raw[section].update(values)
    with pytest.raises(ValueError, match=match):
        parse_sim_config(raw)


@pytest.mark.parametrize("attack, match", [
    ({"type": "sign_flip", "num_malicious": 0}, r"\[1, 10\]"),
    ({"type": "sign_flip", "num_malicious": 10}, "at least one client must be honest"),
    ({"type": "none", "num_malicious": 2}, "requires num_malicious = 0"),
    ({"type": "scaling", "num_malicious": 2}, "attack.gamma"),
    ({"type": "scaling", "num_malicious": 2, "gamma": -1.0}, "must be > 0"),
    ({"type": "gaussian", "num_malicious": 2}, "attack.std"),
    ({"type": "alie", "num_malicious": 1, "z": 1.0}, ">= 2 colluding"),
    ({"type": "alie", "num_malicious": 6, "z": "z_max"}, "s = floor"),
    ({"type": "label_flip", "num_malicious": 2, "source_class": 1, "target_class": 1,
      "poison_fraction": 1.0}, "distinct"),
    ({"type": "label_flip", "num_malicious": 2, "source_class": 1, "target_class": 4,
      "poison_fraction": 1.0}, "distinct"),
    ({"type": "label_flip", "num_malicious": 2, "source_class": 1, "target_class": 2,
      "poison_fraction": 1.5}, r"\(0, 1\]"),
    ({"type": "backdoor", "num_malicious": 2, "target_class": 0, "poison_fraction": 0.5,
      "trigger_mode": "global"}, "no trigger is defined"),
    ({"type": "sign_flip", "num_malicious": 2, "gamma": 3.0}, "takes no"),
    ({"type": "sign_flip", "num_malicious": 2, "start_round": 3}, "after the last round"),
    ({"type": "teleport", "num_malicious": 2}, "unknown attack type"),
])
def test_invalid_attack_configs_rejected(attack, match):
    with pytest.raises(ValueError, match=match):
        parse_sim_config(_raw(attack=attack))


@pytest.mark.parametrize("runtime, match", [
    ({"device": "cuda"}, "device 'cpu'"),
    ({"num_workers": 4}, "num_workers 1"),
    ({"torch_threads": 0}, ">= 1"),
    ({"threads": 2}, "unknown key"),
])
def test_invalid_runtime_rejected(runtime, match):
    with pytest.raises(ValueError, match=match):
        parse_sim_config({**_raw(), "runtime": runtime})


def test_torch_threads_default_and_applied(tmp_path):
    assert parse_sim_config(_raw()).torch_threads == 1
    run_simulation({**_raw(rounds=1), "runtime": {"torch_threads": 2}}, tmp_path / "r")
    env = json.loads((tmp_path / "r" / "config.json").read_text())["environment"]
    assert env["torch_threads"] == 2


def test_protocol_requires_crypto_parameters():
    raw = _raw(protocol="fte")
    del raw["crypto"]["chunk_size"]
    with pytest.raises(ValueError, match="crypto.chunk_size"):
        parse_sim_config(raw)


# ─── Stage A: sampling, seeds, data ──────────────────────────────────────────

def test_client_sampling_is_deterministic_and_valid():
    cfg = parse_sim_config(_raw(k=4, num_clients=10, rounds=5))
    rounds = [sample_participants(cfg, r) for r in range(1, 6)]
    assert rounds == [sample_participants(cfg, r) for r in range(1, 6)]
    for p in rounds:
        assert len(p) == 4 == len(set(p)) and p == sorted(p) and all(0 <= c < 10 for c in p)
    assert len({tuple(p) for p in rounds}) > 1        # varies across rounds
    other = parse_sim_config({**_raw(k=4, rounds=5), "project": {"seed": 8}})
    assert [sample_participants(other, r) for r in range(1, 6)] != rounds


def _strat(mpr=2, num_mal=4, **attack):
    raw = _raw(k=6, rounds=5, num_clients=12,
               attack={"type": "sign_flip", "num_malicious": num_mal, **attack},
               sampling="stratified", malicious_per_round=mpr)
    return raw


def test_stratified_sampling_fixes_malicious_count_per_round():
    cfg = parse_sim_config(_strat())
    mal = fl_attacks.choose_malicious(12, cfg.attack)
    for r in range(1, 6):
        p = sample_participants(cfg, r, mal)
        assert p == sample_participants(cfg, r, mal) == sorted(p) and len(set(p)) == 6
        assert sum(c in mal for c in p) == 2
    assert len({tuple(sample_participants(cfg, r, mal)) for r in range(1, 6)}) > 1
    with pytest.raises(ValueError, match="malicious client set"):
        sample_participants(cfg, 1, [])


def test_uniform_sampling_unchanged_by_malicious_argument():
    cfg = parse_sim_config(_raw(k=4))
    assert sample_participants(cfg, 3, [1, 2]) == sample_participants(cfg, 3)


@pytest.mark.parametrize("raw, match", [
    (_strat(mpr=5), "must be <= num_malicious"),
    (_strat(mpr=6, num_mal=8), "< clients_per_round"),
    (_strat(mpr=1, num_mal=10), "not enough honest clients"),
    ({**_raw(sampling="stratified", malicious_per_round=1)}, "needs malicious clients"),
    (_raw(malicious_per_round=1), "requires sampling 'stratified'"),
    (_raw(sampling="random"), "'uniform' or 'stratified'"),
])
def test_invalid_sampling_configs_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_sim_config(raw)


def test_alie_under_stratified_sampling_checks_per_round_count():
    raw = _strat(mpr=1)
    raw["attack"] = {"type": "alie", "num_malicious": 4, "z": 1.0}
    with pytest.raises(ValueError, match="malicious_per_round >= 2"):
        parse_sim_config(raw)
    raw = _strat(mpr=3)
    raw["attack"] = {"type": "alie", "num_malicious": 4, "z": "z_max"}
    parse_sim_config(raw)                       # n=6, m=3: s = 4 - 3 = 1, valid
    raw = _strat(mpr=4)
    raw["attack"] = {"type": "alie", "num_malicious": 4, "z": "z_max"}
    with pytest.raises(ValueError, match="s = floor"):        # n=6, m=4: s = 0
        parse_sim_config(raw)


def test_stratified_run_records_fixed_malicious_count(tmp_path):
    run_simulation(_strat(), tmp_path / "r")
    assert all(r["filtering"]["malicious_participants"] == 2 for r in _metrics(tmp_path / "r"))


def test_derive_seed_depends_on_every_key():
    base = derive_seed(1, 2, 3)
    assert base == derive_seed(1, 2, 3)
    assert len({base, derive_seed(1, 2, 4), derive_seed(1, 3, 3), derive_seed(2, 2, 3)}) == 4
    assert 0 <= base < 2**64


def test_synthetic_data_is_deterministic_and_held_out_is_separate():
    cfg = parse_sim_config(_raw())
    a, b = build_federated_data(cfg), build_federated_data(cfg)
    assert all(torch.equal(x.x, y.x) and torch.equal(x.y, y.y) for x, y in zip(a.clients, b.clients))
    assert a.info["partition_id"] == b.info["partition_id"]
    train_rows = {tuple(r.tolist()) for c in a.clients for r in c.x}
    assert not any(tuple(r.tolist()) in train_rows for r in a.test[0])
    assert a.validation is None and a.info["test_samples"] == 200


def test_max_eval_samples_takes_a_seeded_subset():
    cfg = parse_sim_config(_raw(max_eval_samples=50))
    full = build_federated_data(parse_sim_config(_raw()))
    sub = build_federated_data(cfg)
    assert sub.test[1].numel() == 50 and sub.info["max_eval_samples"] == 50
    full_rows = {tuple(r.tolist()) for r in full.test[0]}
    assert all(tuple(r.tolist()) in full_rows for r in sub.test[0])


def test_unknown_dataset_key_rejected():
    raw = _raw()
    raw["datasets"]["synthetic"]["noise"] = 1
    with pytest.raises(ValueError, match="unknown key"):
        build_federated_data(parse_sim_config(raw))


# ─── Stage A: update-vector convention ───────────────────────────────────────

class _BNNet(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.lin = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4, momentum=momentum)
        self.frozen = nn.Linear(4, 2)
        for p in self.frozen.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.frozen(self.bn(self.lin(x)))


def test_vector_layout_params_then_float_buffers():
    net = _BNNet()
    assert vector_names(net) == ["lin.weight", "lin.bias", "bn.weight", "bn.bias",
                                 "bn.running_mean", "bn.running_var"]
    assert vector_size(net) == 12 + 4 + 4 + 4 + 4 + 4
    v = get_vector(net)
    assert v.dtype == np.float64
    np.testing.assert_array_equal(v[:12], net.lin.weight.detach().numpy().ravel())
    np.testing.assert_array_equal(v[-4:], np.ones(4))      # running_var init


def test_set_vector_round_trip_and_excluded_tensors_untouched():
    net = _BNNet()
    frozen_before = copy.deepcopy(net.frozen.state_dict())
    tracked_before = net.bn.num_batches_tracked.clone()
    v = np.random.default_rng(0).normal(size=vector_size(net)).astype(np.float32).astype(np.float64)
    set_vector(net, v)
    np.testing.assert_array_equal(get_vector(net), v)
    assert all(torch.equal(frozen_before[k], net.frozen.state_dict()[k]) for k in frozen_before)
    assert torch.equal(net.bn.num_batches_tracked, tracked_before)


def test_set_vector_rejects_bad_input():
    net = _BNNet()
    with pytest.raises(ValueError, match="model expects"):
        set_vector(net, np.zeros(vector_size(net) + 1))
    bad = np.zeros(vector_size(net))
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        set_vector(net, bad)


def test_cumulative_batchnorm_is_rejected():
    with pytest.raises(ValueError, match="momentum=None"):
        get_vector(_BNNet(momentum=None))


def test_copy_excluded_restores_frozen_and_integer_state():
    src, dst = _BNNet(), _BNNet()
    dst.bn.num_batches_tracked += 5
    copy_excluded(src, dst)
    assert torch.equal(dst.frozen.weight, src.frozen.weight)
    assert int(dst.bn.num_batches_tracked) == int(src.bn.num_batches_tracked)
    assert not torch.equal(dst.lin.weight, src.lin.weight)     # vector tensors are not copied


def test_local_training_never_changes_frozen_parameters():
    net = _BNNet()
    frozen = net.frozen.weight.clone()
    cfg = parse_sim_config(_raw())
    client = ClientData("c", torch.randn(16, 3), torch.randint(0, 2, (16,)))
    local_train(net, client, cfg, seed=1)
    assert torch.equal(net.frozen.weight, frozen)


# ─── Stage B: honest FedAvg baseline ─────────────────────────────────────────

@pytest.mark.parametrize("weighting", ["samples", "uniform"])
def test_fedavg_round_equals_weighted_mean_of_client_updates(tmp_path, weighting):
    raw = _raw(rounds=1, k=4, defense={"name": "fedavg", "fedavg_weighting": weighting})
    sim = Simulation(raw, tmp_path / "run")
    sim.setup(sim._prepare_dir())
    part = sample_participants(sim.cfg, 1)
    for ci in part:                      # unequal sample counts so the weightings differ
        c = sim.data.clients[ci]
        sim.data.clients[ci] = ClientData(c.client_id, c.x[:5 + ci], c.y[:5 + ci])
    initial = copy.deepcopy(sim.model)
    g = sim.global_vec.copy()
    rec = sim.run_round(1)

    # recompute independently: same initial model, same per-client seeds
    deltas, counts = [], []
    for ci in part:
        model = copy.deepcopy(initial)
        local_train(model, sim.data.clients[ci], sim.cfg, derive_seed(7, TRAINING, 1, ci))
        deltas.append(get_vector(model) - g)
        counts.append(len(sim.data.clients[ci]))
    assert len(set(counts)) == 4
    w = np.array(counts, float) / sum(counts) if weighting == "samples" else np.full(4, 0.25)
    np.testing.assert_allclose(rec["aggregation_weights"], w, rtol=0, atol=1e-15)
    expected = (g + w @ np.stack(deltas)).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(sim.global_vec, expected)
    # the evaluated model is exactly the stored global vector
    np.testing.assert_array_equal(get_vector(sim.model), sim.global_vec)


def test_evaluation_uses_the_held_out_test_set(tmp_path, monkeypatch):
    seen = []
    real = sim_mod.metrics.evaluate

    def spy(model, x, y, bs):
        seen.append((x, y))
        return real(model, x, y, bs)

    monkeypatch.setattr(sim_mod.metrics, "evaluate", spy)
    sim = Simulation(_raw(rounds=1), tmp_path / "run")
    sim.run()
    assert len(seen) == 1                                    # synthetic: no validation set
    assert seen[0][0] is sim.data.test[0] and seen[0][1] is sim.data.test[1]
    assert _metrics(tmp_path / "run")[0]["validation"] is None


def test_runs_are_bitwise_reproducible(tmp_path):
    raw = _raw(rounds=3, k=6, defense={"name": "multi_krum", "f": 1},
               attack={"type": "gaussian", "num_malicious": 2, "std": 0.5, "seed": 3})
    run_simulation(raw, tmp_path / "a")
    run_simulation(copy.deepcopy(raw), tmp_path / "b")
    ma, mb = _metrics(tmp_path / "a"), _metrics(tmp_path / "b")
    assert [_strip_times(r) for r in ma] == [_strip_times(r) for r in mb]
    va = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)["vector"]
    vb = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)["vector"]
    assert torch.equal(va, vb)


def test_different_seed_changes_the_run(tmp_path):
    raw = _raw(rounds=1)
    run_simulation(raw, tmp_path / "a")
    run_simulation({**raw, "project": {"seed": 8}}, tmp_path / "b")
    assert _metrics(tmp_path / "a")[0]["test"] != _metrics(tmp_path / "b")[0]["test"]


def test_resume_after_failure_matches_uninterrupted_run(tmp_path, monkeypatch):
    raw = _raw(rounds=4, k=6, defense={"name": "krum", "f": 1})
    run_simulation(raw, tmp_path / "full")

    real = Simulation.run_round

    def crash_in_round_3(self, r):
        if r == 3:
            raise RuntimeError("simulated crash")
        return real(self, r)

    monkeypatch.setattr(Simulation, "run_round", crash_in_round_3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_simulation(raw, tmp_path / "part")
    summary = json.loads((tmp_path / "part" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["completed_rounds"] == 2
    assert "simulated crash" in summary["error"]

    monkeypatch.setattr(Simulation, "run_round", real)
    out = run_simulation(raw, tmp_path / "part", resume=True)
    assert out["status"] == "completed" and out["completed_rounds"] == 4
    full, part = _metrics(tmp_path / "full"), _metrics(tmp_path / "part")
    assert [r["round"] for r in part] == [1, 2, 3, 4]
    assert [_strip_times(r) for r in full] == [_strip_times(r) for r in part]
    assert torch.equal(torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)["vector"],
                       torch.load(tmp_path / "part" / "checkpoint.pt", weights_only=False)["vector"])


def test_resume_drops_metrics_written_after_the_checkpoint(tmp_path):
    raw = _raw(rounds=3)
    run_simulation(raw, tmp_path / "r")
    ckpt = torch.load(tmp_path / "r" / "checkpoint.pt", weights_only=False)
    ckpt["round"] = 2       # as if the process died after writing round 3's metrics line
    torch.save(ckpt, tmp_path / "r" / "checkpoint.pt")
    run_simulation(raw, tmp_path / "r", resume=True)
    # round 3's stale line was dropped and written once by the resumed run
    assert [m["round"] for m in _metrics(tmp_path / "r")] == [1, 2, 3]
    # crash after round 1's metrics line but before its first checkpoint
    run_simulation(_raw(rounds=1), tmp_path / "s")
    first = _metrics(tmp_path / "s")
    (tmp_path / "s" / "checkpoint.pt").unlink()
    run_simulation(_raw(rounds=1), tmp_path / "s", resume=True)
    assert [_strip_times(m) for m in _metrics(tmp_path / "s")] == [_strip_times(m) for m in first]


def test_existing_run_dir_is_never_overwritten(tmp_path):
    raw = _raw(rounds=1)
    run_simulation(raw, tmp_path / "r")
    before = (tmp_path / "r" / "summary.json").read_text()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_simulation(raw, tmp_path / "r")
    assert (tmp_path / "r" / "summary.json").read_text() == before


def test_resume_with_a_different_config_is_rejected(tmp_path):
    run_simulation(_raw(rounds=1), tmp_path / "r")
    before = (tmp_path / "r" / "summary.json").read_text()
    with pytest.raises(ValueError, match="config hash"):
        run_simulation(_raw(rounds=2), tmp_path / "r", resume=True)
    assert (tmp_path / "r" / "summary.json").read_text() == before


def test_run_records_reproducibility_metadata(tmp_path):
    run_simulation(_raw(rounds=1), tmp_path / "r")
    cfg = json.loads((tmp_path / "r" / "config.json").read_text())
    for key in ("config_hash", "raw", "parsed", "environment", "model", "threat_model"):
        assert key in cfg
    env = cfg["environment"]
    assert env["packages"]["torch"] and env["packages"]["tenseal"] == "0.3.16" and env["python"]
    assert len(env["git_commit"]) == 40
    assert cfg["parsed"]["seed"] == 7 and cfg["parsed"]["defense"]["name"] == "fedavg"
    assert cfg["model"]["aggregated_vector_size"] == 8 * 16 + 16 + 16 * 4 + 4
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["data_info"]["num_clients"] == 10 and manifest["malicious_clients"] == []


# ─── Stage C: every defense end to end ───────────────────────────────────────

@pytest.mark.parametrize("defense, k", [
    ({"name": "fedavg"}, 7), ({"name": "krum", "f": 1}, 7), ({"name": "multi_krum", "f": 1, "m": 3}, 7),
    ({"name": "trimmed_mean", "b": 2}, 7), ({"name": "bulyan", "f": 1}, 7),
])
def test_each_defense_runs_and_matches_its_rule(tmp_path, monkeypatch, defense, k):
    captured = {}
    real = sim_mod.aggregate

    def spy(deltas, counts, cfg, r, kh, submissions=None):
        res = real(deltas, counts, cfg, r, kh)
        captured.setdefault(r, (deltas.copy(), res))
        return res

    monkeypatch.setattr(sim_mod, "aggregate", spy)
    run_simulation(_raw(rounds=2, k=k, defense=defense), tmp_path / "r")
    params = {kk: v for kk, v in defense.items() if kk != "name"}
    if defense["name"] == "fedavg":
        params = {"weights": np.full(k, 20.0)}
    for deltas, res in captured.values():
        np.testing.assert_allclose(res.aggregate, combine(deltas, RULES[defense["name"]](deltas, **params)),
                                   rtol=0, atol=1e-15)
    m = _metrics(tmp_path / "r")
    assert len(m) == 2 and all(np.isfinite(r["test"]["loss"]) for r in m)
    if defense["name"] == "multi_krum":
        assert all(sum(w > 0 for w in r["aggregation_weights"]) == 3 for r in m)


# ─── Stage D: EtF and FTE ────────────────────────────────────────────────────

@pytest.mark.parametrize("protocol, defense", [
    ("etf", {"name": "fedavg"}), ("etf", {"name": "multi_krum", "f": 1}),
    ("fte", {"name": "fedavg"}), ("fte", {"name": "trimmed_mean", "b": 1}),
])
def test_encrypted_protocols_reconstruct_the_plaintext_aggregate(protocol, defense):
    cfg = parse_sim_config(_raw(k=5, defense=defense, protocol=protocol))
    from fte_fl.crypto.ckks_manager import CKKSManager
    from fte_fl.defenses.protocols import KeyHolder
    kh = KeyHolder(CKKSManager(**SMALL_CKKS))
    rng = np.random.default_rng(0)
    deltas = rng.normal(size=(5, 212)) * 0.1
    deltas[4] += 3.0                                     # a clear outlier for the filters
    counts = np.array([10, 20, 30, 40, 50])
    enc = aggregate(deltas, counts, cfg, 1, kh)
    plain = aggregate(deltas, counts, parse_sim_config(_raw(k=5, defense=defense)), 1)
    np.testing.assert_array_equal(enc.weights, plain.weights)        # same selection
    np.testing.assert_allclose(enc.aggregate, plain.aggregate, rtol=0, atol=1e-5)
    assert enc.ckks_max_abs_error < 1e-5
    if defense["name"] == "fedavg":
        # weights applied once: the sample-weighted mean, not weighted twice
        np.testing.assert_allclose(plain.aggregate, counts / counts.sum() @ deltas, atol=1e-15)
    if defense["name"] == "multi_krum":
        assert enc.weights[4] == 0.0
    assert enc.ciphertexts["encrypted_aggregate"] == 1        # 212 coords fit in one 4096 chunk
    assert enc.traffic["encrypted_aggregate"] > 0
    assert enc.timings["key_holder_decrypt_s"] > 0
    if protocol == "etf":
        assert enc.ciphertexts["ciphertext_update"] == 5
        if defense["name"] != "fedavg":
            assert enc.ciphertexts["encrypted_squared_distances"] == 10
    else:
        assert enc.traffic["plaintext_update"] == 5 * 212 * 8
        assert enc.ciphertexts["ciphertext_contribution"] == int(np.sum(enc.kept_fraction > 0))


def test_fte_mask_rules_apply_masks_once():
    cfg = parse_sim_config(_raw(k=7, defense={"name": "bulyan", "f": 1}, protocol="fte"))
    from fte_fl.crypto.ckks_manager import CKKSManager
    from fte_fl.defenses.protocols import KeyHolder
    kh = KeyHolder(CKKSManager(**SMALL_CKKS))
    deltas = np.random.default_rng(1).normal(size=(7, 50))
    res = aggregate(deltas, np.ones(7), cfg, 1, kh)
    np.testing.assert_allclose(res.aggregate, combine(deltas, RULES["bulyan"](deltas, f=1)), atol=1e-5)
    assert all(0 < kf < 1 for kf in res.kept_fraction if kf > 0)


@pytest.mark.parametrize("protocol, defense", [("etf", {"name": "krum", "f": 1}),
                                               ("fte", {"name": "bulyan", "f": 1})])
def test_encrypted_protocol_end_to_end_run(tmp_path, protocol, defense):
    run_simulation(_raw(rounds=2, k=7, defense=defense, protocol=protocol), tmp_path / "r")
    cfg = json.loads((tmp_path / "r" / "config.json").read_text())
    assert cfg["crypto"]["ciphertexts_per_update"] == 1
    for r in _metrics(tmp_path / "r"):
        assert r["ckks_max_abs_error"] < 1e-4
        assert r["payload_bytes"]["encrypted_aggregate"] > 0


# ─── Stage E: attestation ────────────────────────────────────────────────────

def test_attested_run_end_to_end(tmp_path):
    run_simulation(_raw(rounds=2, k=5, defense={"name": "krum", "f": 1}, protocol="fte_attested"),
                   tmp_path / "r")
    for r in _metrics(tmp_path / "r"):
        assert r["attestation_rejected_indices"] == []
        assert r["payload_bytes"]["commitment_and_proof"] > 0
        assert r["time_s"]["protocol_steps"]["client_attest_s"] > 0


def test_attested_aggregation_excludes_tampered_submission():
    cfg = parse_sim_config(_raw(k=5, defense={"name": "fedavg"}, protocol="fte_attested"))
    from fte_fl.crypto.ckks_manager import CKKSManager
    from fte_fl.defenses.protocols import AttestedSubmission, KeyHolder
    kh = KeyHolder(CKKSManager(**SMALL_CKKS))
    deltas = np.random.default_rng(2).normal(size=(5, 30))
    subs = [attest_update(deltas[i], 3, i) for i in range(5)]
    s = subs[2]
    subs[2] = AttestedSubmission(s.update + 1.0, s.commitment, s.opening, s.proof)   # not committed
    counts = np.array([1, 2, 3, 4, 5])
    res = aggregate(deltas, counts, cfg, 3, kh, submissions=subs)
    assert res.attestation_rejected.tolist() == [2]
    assert res.weights[2] == 0 and res.kept_fraction[2] == 0
    keep = [0, 1, 3, 4]
    w = counts[keep] / counts[keep].sum()                  # FedAvg over accepted clients only
    np.testing.assert_allclose(res.weights[keep], w, atol=1e-15)
    np.testing.assert_allclose(res.aggregate, w @ deltas[keep], atol=1e-5)
    # a submission attested for another round is rejected too
    subs = [attest_update(deltas[i], 3, i) for i in range(5)]
    subs[0] = attest_update(deltas[0], 4, 0)
    assert aggregate(deltas, counts, cfg, 3, kh, submissions=subs).attestation_rejected.tolist() == [0]


# ─── Stage F: attacks ────────────────────────────────────────────────────────

def _atk(**kw):
    return AttackConfig(**{"num_malicious": 2, "seed": 5, **kw})


def test_choose_malicious_is_deterministic():
    a = fl_attacks.choose_malicious(20, _atk(type="sign_flip", num_malicious=4))
    assert a == fl_attacks.choose_malicious(20, _atk(type="sign_flip", num_malicious=4))
    assert len(a) == 4 == len(set(a)) and a == sorted(a)
    assert a != fl_attacks.choose_malicious(20, _atk(type="sign_flip", num_malicious=4, seed=6))


@pytest.mark.parametrize("attack, check", [
    (_atk(type="sign_flip"), lambda new, old: np.array_equal(new, -old)),
    (_atk(type="scaling", gamma=3.0), lambda new, old: np.allclose(new, 3.0 * old, rtol=0, atol=0)),
    (_atk(type="zero"), lambda new, old: not new.any()),
    (_atk(type="gaussian", std=2.0), lambda new, old: abs(new.std() - 2.0) < 0.3),
])
def test_update_attacks_change_only_malicious_rows(attack, check):
    deltas = np.random.default_rng(3).normal(size=(5, 400))
    mal = np.array([False, True, False, True, False])
    out = fl_attacks.attack_updates(deltas, mal, [0, 4, 6, 8, 9], 2, attack)
    np.testing.assert_array_equal(out[~mal], deltas[~mal])
    assert all(check(out[i], deltas[i]) for i in np.flatnonzero(mal))
    assert not np.shares_memory(out, deltas)


def test_gaussian_attack_is_seeded_per_round_and_client():
    deltas = np.zeros((3, 50))
    mal = np.array([True, True, False])
    a = fl_attacks.attack_updates(deltas, mal, [1, 2, 3], 1, _atk(type="gaussian", std=1.0))
    b = fl_attacks.attack_updates(deltas, mal, [1, 2, 3], 1, _atk(type="gaussian", std=1.0))
    c = fl_attacks.attack_updates(deltas, mal, [1, 2, 3], 2, _atk(type="gaussian", std=1.0))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1]) and not np.array_equal(a[0], c[0])


def test_alie_uses_only_malicious_updates():
    rng = np.random.default_rng(4)
    deltas = rng.normal(size=(6, 30))
    mal = np.array([True, False, True, False, True, False])
    out = fl_attacks.attack_updates(deltas, mal, list(range(6)), 1, _atk(type="alie", z=1.5))
    mu, sd = deltas[mal].mean(axis=0), deltas[mal].std(axis=0, ddof=1)
    for i in np.flatnonzero(mal):
        np.testing.assert_allclose(out[i], mu - 1.5 * sd, atol=1e-12)
    np.testing.assert_array_equal(out[~mal], deltas[~mal])
    changed = deltas.copy()
    changed[~mal] += 100.0                       # honest updates are not read
    out2 = fl_attacks.attack_updates(changed, mal, list(range(6)), 1, _atk(type="alie", z=1.5))
    np.testing.assert_array_equal(out2[mal], out[mal])
    # z_max uses n participants and m malicious participants
    from fte_fl.attacks.updates import alie_z_max
    out3 = fl_attacks.attack_updates(deltas, mal, list(range(6)), 1, _atk(type="alie", z="z_max"))
    np.testing.assert_allclose(out3[0], mu - alie_z_max(6, 3) * sd, atol=1e-12)


def test_alie_with_fewer_than_two_malicious_participants_fails():
    with pytest.raises(ValueError, match=">= 2 malicious participants"):
        fl_attacks.attack_updates(np.ones((4, 3)), np.array([1, 0, 0, 0], bool), [0, 1, 2, 3], 1,
                                  _atk(type="alie", z=1.0))


def test_label_flip_poisons_only_selected_source_samples():
    y = torch.tensor([1, 2, 1, 1, 0, 1, 3, 1])
    client = ClientData("c", torch.randn(8, 4), y)
    atk = _atk(type="label_flip", source_class=1, target_class=3, poison_fraction=0.6)
    poisoned, rec = fl_attacks.poison_client(client, 0, 7, "synthetic", 4, atk)
    changed = torch.nonzero(poisoned.y != y).flatten().tolist()
    assert rec == {"client_id": "c", "eligible": 5, "poisoned": 3} and len(changed) == 3
    assert all(y[i] == 1 and poisoned.y[i] == 3 for i in changed)
    assert torch.equal(poisoned.x, client.x) and torch.equal(client.y, y)    # input untouched
    again, _ = fl_attacks.poison_client(client, 0, 7, "synthetic", 4, atk)
    assert torch.equal(again.y, poisoned.y)


def test_backdoor_poisoning_uses_the_clients_trigger_part():
    x = torch.rand(10, 1, 28, 28) * 0.5 + 0.25
    client = ClientData("w", x, torch.zeros(10, dtype=torch.int64) + 5)
    atk = _atk(type="backdoor", target_class=9, poison_fraction=0.3, trigger_mode="distributed")
    poisoned, rec = fl_attacks.poison_client(client, 2, 4, "femnist", 62, atk)
    assert rec["poisoned"] == 3 and rec["trigger_parts"] == [2]
    rows = torch.nonzero(poisoned.y == 9).flatten()
    assert len(rows) == 3
    part = triggers.image_trigger_mask("femnist", [2])
    for i in range(10):
        if i in rows:
            assert torch.all(poisoned.x[i, 0][part] == 0.0)
            assert torch.equal(poisoned.x[i, 0][~part], x[i, 0][~part])
        else:
            assert torch.equal(poisoned.x[i], x[i]) and poisoned.y[i] == 5
    glob, rec = fl_attacks.poison_client(client, 2, 4, "femnist", 62,
                                         _atk(type="backdoor", target_class=9, poison_fraction=0.3,
                                              trigger_mode="global"))
    assert rec["trigger_parts"] == "all"


def test_backdoor_and_label_flip_eval_sets_are_separate_from_clean_test(tmp_path):
    import dataclasses
    sim = Simulation(_raw(), tmp_path / "r")
    x = torch.rand(12, 1, 28, 28) * 0.5 + 0.25
    y = torch.tensor([0, 3, 3, 1, 0, 2, 3, 0, 1, 1, 2, 0])
    sim.data = type("D", (), {"test": (x, y)})()
    x_before = x.clone()
    sim.cfg = dataclasses.replace(sim.cfg, dataset="femnist", attack=AttackConfig(
        type="backdoor", num_malicious=1, target_class=0, poison_fraction=0.5,
        trigger_mode="distributed"))
    sim._build_attack_eval()
    full = triggers.image_trigger_mask("femnist", None)
    assert sim.backdoor_eval.shape[0] == int((y != 0).sum()) == 8      # target class excluded
    assert torch.all(sim.backdoor_eval[:, 0][:, full] == 0.0)          # full (global) trigger
    assert torch.equal(sim.backdoor_eval[:, 0][:, ~full], x[y != 0][:, 0][:, ~full])
    assert torch.equal(x, x_before) and sim.flip_eval is None          # clean set untouched
    sim.cfg = dataclasses.replace(sim.cfg, attack=AttackConfig(
        type="label_flip", num_malicious=1, source_class=3, target_class=1, poison_fraction=1.0))
    sim._build_attack_eval()
    assert torch.equal(sim.flip_eval, x[y == 3]) and sim.backdoor_eval is None


def test_attack_affects_only_malicious_clients_in_simulation(tmp_path, monkeypatch):
    """Honest participants' updates are identical with and without the attack."""
    captured = {}
    real = sim_mod.aggregate

    def spy(deltas, counts, cfg, r, kh, submissions=None):
        captured.setdefault(cfg.attack.type, {})[r] = deltas.copy()
        return real(deltas, counts, cfg, r, kh)

    monkeypatch.setattr(sim_mod, "aggregate", spy)
    base = dict(k=10, defense={"name": "fedavg"}, rounds=2)
    run_simulation(_raw(**base), tmp_path / "clean")
    attack = {"type": "label_flip", "num_malicious": 3, "source_class": 0, "target_class": 1,
              "poison_fraction": 1.0, "start_round": 2, "seed": 1}
    run_simulation(_raw(**base, attack=attack), tmp_path / "atk")
    mal = fl_attacks.choose_malicious(10, parse_sim_config(_raw(**base, attack=attack)).attack)
    honest = [i for i in range(10) if i not in mal]
    clean, atk = captured["none"], captured["label_flip"]
    # round 1 precedes start_round: everybody honest; round 1 is identical.
    np.testing.assert_array_equal(clean[1], atk[1])
    # round 2: the global models differ only through round-1 aggregation (identical), so
    # honest rows match exactly and malicious rows differ.
    np.testing.assert_array_equal(clean[2][honest], atk[2][honest])
    assert all(not np.array_equal(clean[2][i], atk[2][i]) for i in mal)
    m = _metrics(tmp_path / "atk")
    assert [r["attack_active"] for r in m] == [False, True]
    assert "label_flip" in m[1] and 0 <= m[1]["label_flip"]["flip_rate"] <= 1
    assert m[1]["filtering"]["malicious_participants"] == 3


def test_filtering_metrics_definitions():
    mal = np.array([True, True, False, False, False])
    kept = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    w = np.array([0.0, 1 / 3, 0.0, 1 / 3, 1 / 3])
    m = filtering_metrics(mal, w, kept, assumed_f=1)
    assert (m["malicious_rejected"], m["malicious_selected"], m["honest_rejected"]) == (1, 1, 1)
    assert m["tpr"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 1 / 3 and m["precision"] == 0.5
    assert m["malicious_effective_weight"] == pytest.approx(1 / 3) and m["exceeds_assumed_f"]
    none = filtering_metrics(np.zeros(3, bool), np.ones(3) / 3, np.ones(3), None)
    assert none["tpr"] is None and none["precision"] is None and none["exceeds_assumed_f"] is None
