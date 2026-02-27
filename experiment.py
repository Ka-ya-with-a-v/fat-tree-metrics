import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from models import FatTreeMetrics
from metrics import compute_fat_tree_metrics


@dataclass
class ScalingExperiment:
    k_values: List[int]
    link_capacity_gbps: float = 10.0
    results: List[FatTreeMetrics] = field(default_factory=list)

    def run(self) -> "ScalingExperiment":
        self.results = [
            compute_fat_tree_metrics(k, self.link_capacity_gbps)
            for k in self.k_values
        ]
        return self

    def to_json(self, filepath: Optional[Path] = None) -> str:
        data = [r.to_dict() for r in self.results]
        json_str = json.dumps(data, indent=2)
        if filepath:
            filepath.write_text(json_str)
        return json_str

    def to_csv(self, filepath: Path) -> None:
        if not self.results:
            raise ValueError("No results to export. Run experiment first.")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

    def summary_table(self) -> str:
        if not self.results:
            return "No results. Run experiment first."

        header = f"{'k':>4} | {'Hosts':>10} | {'Switches':>10} | {'Bisection BW':>14} | {'Paths':>6}"
        separator = "-" * len(header)

        rows = [header, separator]
        for r in self.results:
            rows.append(
                f"{r.k:>4} | {r.hosts:>10,} | {r.total_switches:>10,} | "
                f"{r.bisection_bandwidth_gbps:>11,.0f} Gbps | {r.interpod_paths:>6}"
            )

        return "\n".join(rows)
