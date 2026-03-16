import argparse
import glob
import json
import math
import os


ROUND_DIGITS = 6
DEFAULT_LATENCY_KEYS = (
    "average_latency",
    "initial_base_latency",
    "final_base_latency",
)


def to_float_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def load_route_records(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, "*.json"))
    merged_records = []

    for file_path in file_paths:
        if os.path.basename(file_path) == "merged.json":
            continue
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
        records = data.get("_checkpoint", {}).get("records", [])
        for record in records:
            if record.get("status") == "Failed - Agent crashed":
                continue
            clean_record = dict(record)
            clean_record.pop("index", None)
            merged_records.append(clean_record)
    return merged_records


def is_success_record(record):
    if record.get("status") not in {"Completed", "Perfect"}:
        return False

    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return False

    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if isinstance(value, list):
            if len(value) > 0:
                return False
        elif value:
            return False
    return True


def extract_latency_values(record, latency_keys=None):
    latency_data = record.get("latency")
    if not isinstance(latency_data, dict):
        return {}

    keys = list(latency_keys) if latency_keys is not None else list(latency_data.keys())
    return {key: to_float_or_none(latency_data.get(key)) for key in keys}


def _linear_quantile(sorted_values, quantile):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_numeric(values):
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    sorted_values = sorted(values)
    return {
        "mean": round(sum(sorted_values) / len(sorted_values), ROUND_DIGITS),
        "p50": round(_linear_quantile(sorted_values, 0.50), ROUND_DIGITS),
        "p90": round(_linear_quantile(sorted_values, 0.90), ROUND_DIGITS),
        "p95": round(_linear_quantile(sorted_values, 0.95), ROUND_DIGITS),
        "min": round(sorted_values[0], ROUND_DIGITS),
        "max": round(sorted_values[-1], ROUND_DIGITS),
    }


def _infer_latency_keys(records):
    discovered = set()
    for record in records:
        latency_data = record.get("latency")
        if isinstance(latency_data, dict):
            discovered.update(latency_data.keys())

    keys = [key for key in DEFAULT_LATENCY_KEYS if key in discovered]
    keys.extend(sorted(key for key in discovered if key not in DEFAULT_LATENCY_KEYS))
    return keys


def collect_latency_metrics(records, latency_keys=None, require_success=False):
    records_for_latency = records
    if require_success:
        records_for_latency = [record for record in records if is_success_record(record)]

    record_count = len(records_for_latency)
    keys = list(latency_keys) if latency_keys is not None else _infer_latency_keys(records_for_latency)

    values_by_key = {key: [] for key in keys}
    latency_record_count = 0

    for record in records_for_latency:
        latency_data = record.get("latency")
        if isinstance(latency_data, dict):
            latency_record_count += 1
        parsed_values = extract_latency_values(record, keys)
        for key in keys:
            value = parsed_values.get(key)
            if value is not None:
                values_by_key[key].append(value)

    fields = {}
    for key in keys:
        valid_count = len(values_by_key[key])
        coverage = round(valid_count / record_count, ROUND_DIGITS) if record_count else 0.0
        fields[key] = {
            "valid_count": valid_count,
            "missing_count": record_count - valid_count,
            "coverage": coverage,
            **summarize_numeric(values_by_key[key]),
        }

    return {
        "record_count": record_count,
        "latency_record_count": latency_record_count,
        "fields": fields,
    }


def _collect_driving_metrics(records):
    driving_scores = []
    success_num = 0

    for record in records:
        driving_scores.append(record["scores"]["score_composed"])
        if is_success_record(record):
            success_num += 1
            print(record.get("route_id"))

    eval_num = len(driving_scores)
    driving_score_mean = (sum(driving_scores) / eval_num) if eval_num else 0.0
    success_rate = (success_num / eval_num) if eval_num else 0.0
    return driving_score_mean, success_rate, eval_num


def merge_route_json(folder_path, include_latency=True, latency_keys=None):
    merged_records = load_route_records(folder_path)
    if len(merged_records) != 220:
        print(
            "-----------------------Warning: there are "
            f"{len(merged_records)} routes in your json, which does not equal to 220. "
            "All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!"
        )

    driving_score_mean, success_rate, eval_num = _collect_driving_metrics(merged_records)
    merged_records = sorted(merged_records, key=lambda record: record["route_id"], reverse=True)

    merged_data = {
        "_checkpoint": {
            "records": merged_records,
        },
        "driving score": driving_score_mean,
        "success rate": success_rate,
        "eval num": eval_num,
    }
    if include_latency:
        merged_data["latency_metrics"] = collect_latency_metrics(
            merged_records,
            latency_keys=latency_keys,
            require_success=False,
        )

    with open(os.path.join(folder_path, "merged.json"), "w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=4)

    return merged_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder',
        help='old foo help',
        default='eval_results/Bench2Drive/reproCVPTsimlingo_2025_05_02_04_08_01_simlingo_withaugmentation_seed2/bench2drive/1/res',
    )
    parser.add_argument(
        '--no-latency',
        action='store_true',
        help='Disable latency metric aggregation in merged.json.',
    )
    parser.add_argument(
        '--latency-keys',
        nargs='+',
        default=None,
        help='Optional explicit latency keys to aggregate.',
    )
    args = parser.parse_args()

    if os.path.isdir(args.folder):
        merge_route_json(
            args.folder,
            include_latency=not args.no_latency,
            latency_keys=args.latency_keys,
        )
