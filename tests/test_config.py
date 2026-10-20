"""Tests for the config loader module."""
from pathlib import Path

import numpy as np

from fte_fl.config import Config, config_hash, set_seed


CONFIG_PATH = Path(__file__).resolve().parent.parent / "fte_fl" / "configs" / "default.yaml"


def test_config_loads_yaml():
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.seed == 42
    assert cfg.hash  # non-empty string
    assert len(cfg.hash) == 16
    assert isinstance(cfg.raw, dict)


def test_config_hash_is_deterministic():
    d = {"a": 1, "b": [2, 3]}
    assert config_hash(d) == config_hash(d)


def test_config_hash_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_default_defense_name_is_a_registered_rule():
    """The config uses the registry's canonical names (e.g. 'multi_krum')."""
    from fte_fl.defenses.aggregators import RULES

    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg["defense"]["name"] == "multi_krum"
    assert cfg["defense"]["name"] in RULES


def test_config_to_dict_includes_reproducibility():
    cfg = Config.from_yaml(CONFIG_PATH)
    d = cfg.to_dict()
    assert "config_hash" in d
    assert "git_commit" in d
    assert "seed" in d
