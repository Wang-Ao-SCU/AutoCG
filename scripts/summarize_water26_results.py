#!/usr/bin/env python3
"""Summarize phase-classification errors for solvent 26 water pairs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="phase_separation_results_water26.csv")
    parser.add_argument("--output-csv", default="water26_mismatch_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    with input_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    water_rows = [
        row for row in rows
        if row.get("X") == "26" or row.get("Y") == "26"
    ]
    mismatches = [
        row for row in water_rows
        if (row.get("Z1") or "").strip() in {"0", "1"}
        and (row.get("phase_sep_sim") or "").strip() in {"0", "1"}
        and (row.get("Z1") or "").strip() != (row.get("phase_sep_sim") or "").strip()
    ]
    status_counts = Counter((row.get("status") or "").strip() for row in water_rows)
    confusion = Counter(
        ((row.get("Z1") or "").strip(), (row.get("phase_sep_sim") or "").strip())
        for row in water_rows
        if (row.get("Z1") or "").strip() in {"0", "1"}
        and (row.get("phase_sep_sim") or "").strip() in {"0", "1"}
    )

    out_fields = [
        "X",
        "Y",
        "Name1",
        "Name2",
        "Z1",
        "phase_sep_sim",
        "hetero_neighbor_fraction",
        "status",
        "L2",
        "diagnosis",
    ]

    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for row in mismatches:
            exp = (row.get("Z1") or "").strip()
            sim = (row.get("phase_sep_sim") or "").strip()
            if exp == "0" and sim == "1":
                diagnosis = "sim_overseparates_water_pair"
            elif exp == "1" and sim == "0":
                diagnosis = "sim_overmixes_water_pair"
            else:
                diagnosis = "unclassified"
            writer.writerow({name: row.get(name, "") for name in out_fields[:-1]} | {"diagnosis": diagnosis})

    comparable = sum(confusion.values())
    correct = confusion[("0", "0")] + confusion[("1", "1")]
    accuracy = correct / comparable if comparable else 0.0

    print(f"Water-26 rows: {len(water_rows)}")
    print(f"Comparable water-26 rows: {comparable}")
    print(f"Water-26 accuracy: {correct}/{comparable} = {accuracy:.2%}" if comparable else "Water-26 accuracy: no comparable rows")
    print(f"Confusion exp/sim: {dict(confusion)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Mismatches written: {output_csv} ({len(mismatches)} rows)")


if __name__ == "__main__":
    main()
