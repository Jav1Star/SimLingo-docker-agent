import json

import pytest

from team_code_adaption.behavior_window_logger import BehaviorWindowLogger


def _logger(tmp_path, window_size=50):
    return BehaviorWindowLogger(
        tmp_path / "behaviors" / "009",
        route_id="009",
        route_key="RouteScenario_test",
        prune_ratio=0.25,
        mode="fixed",
        fixed_budget=0.5,
        predict_language_enabled=True,
        predict_language_max_new_tokens=32,
        predict_language_stride=5,
        window_size=window_size,
    )


def test_writes_complete_behavior_window(tmp_path):
    logger = _logger(tmp_path, window_size=3)
    for step, duration in enumerate((0.1, 0.2, 0.3), start=1):
        logger.add_step(
            step=step,
            timestamp=step / 20,
            remote_inference_sec=duration,
            layer_active_mask=[1] * 10 + [0] * 14,
        )

    output = tmp_path / "behaviors" / "009" / "step_0_3.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["route_id"] == "009"
    assert payload["predict_language_stride"] == 5
    assert payload["sample_count"] == 3
    assert payload["window_complete"] is True
    assert payload["avg_remote_inference_sec"] == pytest.approx(0.2)
    assert [item["step"] for item in payload["steps"]] == [1, 2, 3]
    assert all(len(item["layer_active_mask"]) == 24 for item in payload["steps"])


def test_finalize_writes_partial_tail_once(tmp_path):
    logger = _logger(tmp_path, window_size=3)
    logger.add_step(step=8, timestamp=0.4, remote_inference_sec=0.25, layer_active_mask=[1] * 24)
    logger.finalize()
    logger.finalize()

    payload = json.loads(
        (tmp_path / "behaviors" / "009" / "step_0_3.json").read_text(encoding="utf-8")
    )
    assert payload["window_complete"] is False
    assert payload["sample_count"] == 1
    assert payload["first_observed_step"] == 8
