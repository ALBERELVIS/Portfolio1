#!/usr/bin/env python3
"""Search tool for identifying standout assets in a return dataset."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class AssetStats:
    name: str
    mean: float
    volatility: float
    sharpe: float
    minimum: float
    maximum: float
    positive_pct: float
    cvar_95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and rank assets to identify standout behavior."
    )
    parser.add_argument(
        "--data",
        default="prod_long_sharpe_u50_20260116_v5_train_dataset.csv",
        help="Path to the CSV dataset of asset returns.",
    )
    parser.add_argument(
        "--sort",
        default="sharpe",
        choices=[
            "sharpe",
            "mean",
            "volatility",
            "positive_pct",
            "cvar_95",
            "range",
        ],
        help="Metric to sort by.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of assets to show.")
    parser.add_argument("--min-mean", type=float, help="Minimum average return.")
    parser.add_argument("--max-volatility", type=float, help="Maximum volatility.")
    parser.add_argument("--min-sharpe", type=float, help="Minimum Sharpe-like ratio.")
    parser.add_argument(
        "--min-positive",
        type=float,
        help="Minimum percent of positive observations (0-100).",
    )
    parser.add_argument(
        "--name-regex",
        help="Regex to filter asset names (case-insensitive).",
    )
    return parser.parse_args()


def read_returns(path: str) -> dict[str, List[float]]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        series = {name: [] for name in headers}
        for row in reader:
            if not row:
                continue
            for name, value in zip(headers, row):
                series[name].append(float(value))
    return series


def cvar(values: Sequence[float], tail_fraction: float = 0.05) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    tail_size = max(1, int(math.ceil(len(sorted_vals) * tail_fraction)))
    tail = sorted_vals[:tail_size]
    return sum(tail) / len(tail)


def summarize_assets(series: dict[str, List[float]]) -> List[AssetStats]:
    stats: List[AssetStats] = []
    for name, values in series.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = (
            sum((val - mean) ** 2 for val in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        volatility = math.sqrt(variance)
        sharpe = mean / volatility if volatility else 0.0
        minimum = min(values)
        maximum = max(values)
        positive_pct = 100.0 * sum(1 for v in values if v > 0) / len(values)
        stats.append(
            AssetStats(
                name=name,
                mean=mean,
                volatility=volatility,
                sharpe=sharpe,
                minimum=minimum,
                maximum=maximum,
                positive_pct=positive_pct,
                cvar_95=cvar(values),
            )
        )
    return stats


def filter_assets(
    assets: Iterable[AssetStats],
    min_mean: float | None,
    max_volatility: float | None,
    min_sharpe: float | None,
    min_positive: float | None,
    name_regex: str | None,
) -> List[AssetStats]:
    filtered = []
    name_pattern = re.compile(name_regex, re.IGNORECASE) if name_regex else None
    for asset in assets:
        if min_mean is not None and asset.mean < min_mean:
            continue
        if max_volatility is not None and asset.volatility > max_volatility:
            continue
        if min_sharpe is not None and asset.sharpe < min_sharpe:
            continue
        if min_positive is not None and asset.positive_pct < min_positive:
            continue
        if name_pattern and not name_pattern.search(asset.name):
            continue
        filtered.append(asset)
    return filtered


def sort_assets(assets: List[AssetStats], metric: str) -> List[AssetStats]:
    key_map = {
        "sharpe": lambda item: item.sharpe,
        "mean": lambda item: item.mean,
        "volatility": lambda item: item.volatility,
        "positive_pct": lambda item: item.positive_pct,
        "cvar_95": lambda item: item.cvar_95,
        "range": lambda item: item.maximum - item.minimum,
    }
    reverse = metric != "volatility" and metric != "cvar_95"
    return sorted(assets, key=key_map[metric], reverse=reverse)


def format_table(assets: Sequence[AssetStats]) -> str:
    headers = [
        "asset",
        "mean",
        "volatility",
        "sharpe",
        "min",
        "max",
        "% positive",
        "cvar_95",
    ]
    rows = [
        [
            asset.name,
            f"{asset.mean:.6f}",
            f"{asset.volatility:.6f}",
            f"{asset.sharpe:.2f}",
            f"{asset.minimum:.6f}",
            f"{asset.maximum:.6f}",
            f"{asset.positive_pct:.1f}",
            f"{asset.cvar_95:.6f}",
        ]
        for asset in assets
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    series = read_returns(args.data)
    stats = summarize_assets(series)
    filtered = filter_assets(
        stats,
        min_mean=args.min_mean,
        max_volatility=args.max_volatility,
        min_sharpe=args.min_sharpe,
        min_positive=args.min_positive,
        name_regex=args.name_regex,
    )
    ranked = sort_assets(filtered, args.sort)
    top_assets = ranked[: args.top]
    print(format_table(top_assets))


if __name__ == "__main__":
    main()
