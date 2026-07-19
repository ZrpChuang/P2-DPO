import argparse
import json
from pathlib import Path


PPL_KEYS = (
    ("chosen_ppl", "rejected_ppl"),
    ("win_ppl", "lose_ppl"),
    ("yw_ppl", "yl_ppl"),
    ("ppl_chosen", "ppl_rejected"),
    ("ppl_win", "ppl_lose"),
    ("ppl_yw", "ppl_yl"),
)

LOGP_KEYS = (
    ("chosen_logp", "rejected_logp"),
    ("win_logp", "lose_logp"),
    ("yw_logp", "yl_logp"),
    ("chosen_log_prob", "rejected_log_prob"),
    ("win_log_prob", "lose_log_prob"),
    ("yw_log_prob", "yl_log_prob"),
    ("logp_chosen", "logp_rejected"),
    ("logp_win", "logp_lose"),
    ("logp_yw", "logp_yl"),
    ("log_prob_chosen", "log_prob_rejected"),
    ("log_prob_win", "log_prob_lose"),
    ("log_prob_yw", "log_prob_yl"),
)

PAIR_KEYS = ("focus", "robust", "pair", "preference")


def read_records(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()], "jsonl"
    data = json.loads(text)
    if isinstance(data, list):
        return data, "json"
    for key in ("data", "records", "pairs", "examples"):
        if isinstance(data.get(key), list):
            return data[key], ("json", data, key)
    raise ValueError("Input JSON must be a list or contain a list field.")


def write_records(path, records, input_format):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl" or input_format == "jsonl":
        with output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    if isinstance(input_format, tuple):
        _, original, key = input_format
        original[key] = records
        payload = original
    else:
        payload = records
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def as_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_pair_values(record, keys):
    candidates = [record]
    candidates.extend(value for key, value in record.items() if key in PAIR_KEYS and isinstance(value, dict))
    for candidate in candidates:
        for chosen_key, rejected_key in keys:
            chosen = as_number(candidate.get(chosen_key))
            rejected = as_number(candidate.get(rejected_key))
            if chosen is not None and rejected is not None:
                return chosen, rejected
    return None


def keep_record(record, max_ppl, margin_low, margin_high, require_metrics):
    ppl = find_pair_values(record, PPL_KEYS)
    logp = find_pair_values(record, LOGP_KEYS)
    if require_metrics and (ppl is None or logp is None):
        return False, "missing_metrics"
    if ppl is not None and max(ppl) >= max_ppl:
        return False, "ppl"
    if logp is not None:
        margin = logp[0] - logp[1]
        if margin < margin_low or margin > margin_high:
            return False, "margin"
        record["logp_margin"] = margin
    return True, "kept"


def filter_records(records, args):
    kept = []
    stats = {
        "total": 0,
        "kept": 0,
        "missing_metrics": 0,
        "ppl": 0,
        "margin": 0,
    }
    for record in records:
        stats["total"] += 1
        ok, reason = keep_record(
            record,
            max_ppl=args.max_ppl,
            margin_low=args.margin_low,
            margin_high=args.margin_high,
            require_metrics=args.require_metrics,
        )
        stats[reason] += 1
        if ok:
            kept.append(record)
    return kept, stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-ppl", type=float, default=2.5)
    parser.add_argument("--margin-low", type=float, default=-0.174)
    parser.add_argument("--margin-high", type=float, default=0.208)
    parser.add_argument("--require-metrics", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records, input_format = read_records(args.input)
    kept, stats = filter_records(records, args)
    write_records(args.output, kept, input_format)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
