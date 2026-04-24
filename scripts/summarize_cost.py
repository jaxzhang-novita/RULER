#!/usr/bin/env python
# Summarize token usage and estimated cost from RULER prediction JSONL files.

import argparse
import csv
import json
from pathlib import Path


def format_int(value):
    return f"{value:,}"


def format_tokens(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def usage_value(usage, key):
    value = usage.get(key, 0) if usage else 0
    return int(value or 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--input_price", type=float, default=12.0, help="price per 1M input tokens")
    parser.add_argument("--output_price", type=float, default=24.0, help="price per 1M output tokens")
    parser.add_argument("--currency", type=str, default="CNY")
    parser.add_argument("--elapsed_seconds", type=float, default=0.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_samples = 0

    pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
    for path in sorted(args.pred_dir.glob(pattern)):
        prompt_tokens = 0
        completion_tokens = 0
        tokens = 0
        samples = 0
        for record in read_jsonl(path):
            usage = record.get("usage", {})
            if not usage:
                continue
            prompt_tokens += usage_value(usage, "prompt_tokens")
            completion_tokens += usage_value(usage, "completion_tokens")
            tokens += usage_value(usage, "total_tokens")
            samples += 1

        if samples == 0:
            continue

        input_cost = prompt_tokens / 1_000_000 * args.input_price
        output_cost = completion_tokens / 1_000_000 * args.output_price
        cost = input_cost + output_cost
        rows.append({
            "task": str(path.relative_to(args.pred_dir)).removesuffix(".jsonl"),
            "samples": samples,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(cost, 6),
        })
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_tokens += tokens
        total_samples += samples

    output_file = args.pred_dir / "cost_summary.csv"
    with open(output_file, "w", encoding="utf-8", newline="") as fout:
        fieldnames = [
            "task",
            "samples",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_input_cost = total_prompt / 1_000_000 * args.input_price
    total_output_cost = total_completion / 1_000_000 * args.output_price
    total_cost = total_input_cost + total_output_cost

    if args.verbose:
        print("\nCost summary")
        print(f"  samples       : {format_int(total_samples)}")
        print(f"  input tokens  : {format_int(total_prompt)}")
        print(f"  output tokens : {format_int(total_completion)}")
        print(f"  total tokens  : {format_int(total_tokens)}")
        print(f"  input cost    : {total_input_cost:.6f} {args.currency}")
        print(f"  output cost   : {total_output_cost:.6f} {args.currency}")
        print(f"  total cost    : {total_cost:.6f} {args.currency}")
        if args.elapsed_seconds > 0:
            print(f"  elapsed       : {args.elapsed_seconds:.2f}s")
            print(f"  input tps     : {total_prompt / args.elapsed_seconds:.2f}")
            print(f"  output tps    : {total_completion / args.elapsed_seconds:.2f}")
            print(f"  total tps     : {total_tokens / args.elapsed_seconds:.2f}")
        print(f"  saved to      : {output_file}")
        return

    label = f"{args.label}: " if args.label else ""
    parts = [
        f"{label}samples={format_int(total_samples)}",
        f"in={format_tokens(total_prompt)}",
        f"out={format_tokens(total_completion)}",
        f"cost={total_cost:.4f} {args.currency}",
    ]
    if args.elapsed_seconds > 0:
        parts.append(f"time={args.elapsed_seconds:.0f}s")
        parts.append(f"tps={total_tokens / args.elapsed_seconds:.1f}")
    parts.append(f"csv={output_file}")
    print(" | ".join(parts))


if __name__ == "__main__":
    main()
