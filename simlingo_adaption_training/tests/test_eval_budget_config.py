import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from start_eval_simlingo_adaption import build_eval_config, parse_budget_settings


class EvalBudgetConfigTest(unittest.TestCase):
    def _build_min_eval_section(self):
        return {
            "agent": "simlingo",
            "checkpoint": "~/ckpt.bin",
            "benchmark": "bench2drive",
            "route_path": "~/routes",
            "seeds": [3],
            "tries": 0,
            "out_root": "~/out",
            "carla_root": "/data/carla0915",
            "repo_root": "~/repo",
            "agent_file": "~/agent.py",
        }

    def test_rule_based_config_merge_and_build(self):
        full_rule_based_cfg = {
            "k_warmup": 7,
            "eta": 0.03,
            "safe_threshold": 0.35,
            "safe_count_threshold": 2,
            "decay_step": 0.1,
            "base_offset": 0.25,
            "base_scale": 0.75,
            "inst_offset": 0.5,
            "inst_scale": 0.5,
            "weights": {
                "base_mean": {"novelty": 0.23, "speed_shift": 0.35, "route_shift": 0.15},
                "base_max": {"novelty": 0.08, "speed_shift": 0.12, "route_shift": 0.08},
                "inst": {"novelty": 0.5, "speed_shift": 0.40, "route_shift": 0.20},
            },
            "normalization": {
                "novelty": {"q10": 0.0001102686, "q90": 0.0150763988},
                "speed_shift": {"q10": 0.0030981766, "q90": 0.7491058707},
                "route_shift": {"q10": 0.1959435195, "q90": 0.7666570544},
            },
        }
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "budget": {
                "mode": "rule_based",
                "fixed_budget": 0.9,
                "rule_based": full_rule_based_cfg,
            },
        }

        mode, fixed_budget, rule_based_cfg = parse_budget_settings(cfg_yaml)
        self.assertEqual(mode, "rule_based")
        self.assertAlmostEqual(fixed_budget, 0.9)
        self.assertEqual(rule_based_cfg["k_warmup"], 7)
        self.assertAlmostEqual(rule_based_cfg["weights"]["inst"]["novelty"], 0.5)
        self.assertIn("normalization", rule_based_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "eval.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_yaml, f)

            args = SimpleNamespace(eval_config=str(cfg_path), seed=5)
            eval_cfg = build_eval_config(args, no_server_launch=True)

        self.assertEqual(eval_cfg["budget_mode"], "rule_based")
        self.assertEqual(eval_cfg["seeds"], [5])
        self.assertTrue(eval_cfg["no_server_launch"])
        self.assertEqual(eval_cfg["rule_based_cfg"]["k_warmup"], 7)
        self.assertAlmostEqual(eval_cfg["rule_based_cfg"]["weights"]["inst"]["novelty"], 0.5)

    def test_rule_based_requires_config(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "budget": {
                "mode": "rule_based",
                "fixed_budget": 0.9,
            },
        }
        with self.assertRaises(ValueError):
            parse_budget_settings(cfg_yaml)

    def test_fixed_budget_bound_check(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "budget": {
                "mode": "fixed",
                "fixed_budget": 1.5,
            },
        }
        with self.assertRaises(ValueError):
            parse_budget_settings(cfg_yaml)

    def test_random_mode_is_supported(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "budget": {
                "mode": "random",
            },
        }
        mode, _, _ = parse_budget_settings(cfg_yaml)
        self.assertEqual(mode, "random")

    def test_invalid_mode_check(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "budget": {
                "mode": "invalid_mode",
                "fixed_budget": 0.8,
            },
        }
        with self.assertRaises(ValueError):
            parse_budget_settings(cfg_yaml)


if __name__ == "__main__":
    unittest.main()
