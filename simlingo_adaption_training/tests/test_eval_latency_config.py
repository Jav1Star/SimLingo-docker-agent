import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from start_eval_simlingo_adaption import build_eval_config, parse_latency_settings


class EvalLatencyConfigTest(unittest.TestCase):
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
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "latency": {
                "mode": "rule_based",
                "fix_latency": 0.9,
                "rule_based": {
                    "k_warmup": 7,
                    "weights": {
                        "inst": {
                            "novelty": 0.5,
                        }
                    },
                },
            },
        }

        mode, fix_latency, rule_based_cfg = parse_latency_settings(cfg_yaml)
        self.assertEqual(mode, "rule_based")
        self.assertAlmostEqual(fix_latency, 0.9)
        self.assertEqual(rule_based_cfg["k_warmup"], 7)
        self.assertAlmostEqual(rule_based_cfg["weights"]["inst"]["novelty"], 0.5)
        self.assertIn("normalization", rule_based_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "eval.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_yaml, f)

            args = SimpleNamespace(eval_config=str(cfg_path), seed=5)
            eval_cfg = build_eval_config(args, no_server_launch=True)

        self.assertEqual(eval_cfg["latency_mode"], "rule_based")
        self.assertEqual(eval_cfg["seeds"], [5])
        self.assertTrue(eval_cfg["no_server_launch"])
        self.assertEqual(eval_cfg["rule_based_cfg"]["k_warmup"], 7)
        self.assertAlmostEqual(eval_cfg["rule_based_cfg"]["weights"]["inst"]["novelty"], 0.5)

    def test_fixed_latency_bound_check(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "latency": {
                "mode": "fixed",
                "fix_latency": 1.5,
            },
        }
        with self.assertRaises(ValueError):
            parse_latency_settings(cfg_yaml)

    def test_random_mode_is_supported(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "latency": {
                "mode": "random",
            },
        }
        mode, _, _ = parse_latency_settings(cfg_yaml)
        self.assertEqual(mode, "random")

    def test_invalid_mode_check(self):
        cfg_yaml = {
            "eval": self._build_min_eval_section(),
            "latency": {
                "mode": "fix-latency",
                "fix_latency": 0.8,
            },
        }
        with self.assertRaises(ValueError):
            parse_latency_settings(cfg_yaml)


if __name__ == "__main__":
    unittest.main()
