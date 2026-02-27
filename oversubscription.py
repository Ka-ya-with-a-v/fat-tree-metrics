import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from models import OversubscriptionResult
from metrics import validate_k


class FatTreeTopology:
    def __init__(self, k: int, oversubscription_ratio: float = 1.0):
        validate_k(k)
        self.k = k
        self.half_k = k // 2
        self.oversubscription_ratio = oversubscription_ratio

        self.num_pods = k
        self.num_hosts = (k ** 3) // 4
        self.hosts_per_edge = self.half_k
        self.edge_per_pod = self.half_k
        self.agg_per_pod = self.half_k
        self.num_core = self.half_k ** 2

        self.host_location = {}
        host_id = 0
        for pod in range(self.num_pods):
            for edge in range(self.edge_per_pod):
                for h in range(self.hosts_per_edge):
                    self.host_location[host_id] = (pod, edge, h)
                    host_id += 1

        self.effective_uplinks_per_edge = self.half_k / oversubscription_ratio
        self.effective_uplinks_per_agg = self.half_k / oversubscription_ratio

    def get_host_pod(self, host_id: int) -> int:
        return self.host_location[host_id][0]

    def get_host_edge(self, host_id: int) -> Tuple[int, int]:
        loc = self.host_location[host_id]
        return (loc[0], loc[1])

    def get_path_links(self, src_host: int, dst_host: int) -> List[str]:
        src_pod, src_edge, _ = self.host_location[src_host]
        dst_pod, dst_edge, _ = self.host_location[dst_host]

        links = []
        links.append(f"host_{src_host}_edge_{src_pod}_{src_edge}")

        if src_pod == dst_pod and src_edge == dst_edge:
            links.append(f"edge_{dst_pod}_{dst_edge}_host_{dst_host}")
            return links

        src_agg = random.randint(0, self.agg_per_pod - 1)
        links.append(f"edge_{src_pod}_{src_edge}_agg_{src_pod}_{src_agg}")

        if src_pod == dst_pod:
            links.append(f"agg_{dst_pod}_{src_agg}_edge_{dst_pod}_{dst_edge}")
        else:
            core_base = src_agg * self.half_k
            core_offset = random.randint(0, self.half_k - 1)
            core_id = core_base + core_offset

            links.append(f"agg_{src_pod}_{src_agg}_core_{core_id}")

            dst_agg = core_offset
            links.append(f"core_{core_id}_agg_{dst_pod}_{dst_agg}")
            links.append(f"agg_{dst_pod}_{dst_agg}_edge_{dst_pod}_{dst_edge}")

        links.append(f"edge_{dst_pod}_{dst_edge}_host_{dst_host}")
        return links

    def get_link_capacity(self, link_id: str) -> float:
        if "host_" in link_id and "edge_" in link_id:
            return 1.0
        elif "edge_" in link_id and "agg_" in link_id:
            return 1.0 / self.oversubscription_ratio
        elif "agg_" in link_id and "core_" in link_id:
            return 1.0 / self.oversubscription_ratio
        elif "core_" in link_id:
            return 1.0 / self.oversubscription_ratio
        else:
            return 1.0


def simulate_oversubscription(
    k: int,
    oversubscription_ratio: float = 1.0,
    num_flows: int = 100,
    traffic_pattern: str = "random",
    seed: Optional[int] = None,
) -> OversubscriptionResult:
    if seed is not None:
        random.seed(seed)

    topo = FatTreeTopology(k, oversubscription_ratio)
    flows = []
    num_hosts = topo.num_hosts

    if traffic_pattern == "random":
        for _ in range(num_flows):
            src = random.randint(0, num_hosts - 1)
            dst = random.randint(0, num_hosts - 1)
            while dst == src:
                dst = random.randint(0, num_hosts - 1)
            flows.append((src, dst))

    elif traffic_pattern == "all_to_all":
        all_pairs = [(s, d) for s in range(num_hosts) for d in range(num_hosts) if s != d]
        flows = random.sample(all_pairs, min(num_flows, len(all_pairs)))

    elif traffic_pattern == "intra_pod":
        for _ in range(num_flows):
            pod = random.randint(0, topo.num_pods - 1)
            hosts_in_pod = [h for h, loc in topo.host_location.items() if loc[0] == pod]
            src, dst = random.sample(hosts_in_pod, 2)
            flows.append((src, dst))

    elif traffic_pattern == "inter_pod":
        for _ in range(num_flows):
            src_pod = random.randint(0, topo.num_pods - 1)
            dst_pod = random.randint(0, topo.num_pods - 1)
            while dst_pod == src_pod:
                dst_pod = random.randint(0, topo.num_pods - 1)
            src_hosts = [h for h, loc in topo.host_location.items() if loc[0] == src_pod]
            dst_hosts = [h for h, loc in topo.host_location.items() if loc[0] == dst_pod]
            flows.append((random.choice(src_hosts), random.choice(dst_hosts)))
    else:
        raise ValueError(f"Unknown traffic pattern: {traffic_pattern}")

    link_flows: Dict[str, List[int]] = defaultdict(list)
    flow_paths: List[List[str]] = []

    for flow_idx, (src, dst) in enumerate(flows):
        path = topo.get_path_links(src, dst)
        flow_paths.append(path)
        for link in path:
            link_flows[link].append(flow_idx)

    flow_throughput = []
    for flow_idx, path in enumerate(flow_paths):
        min_share = float('inf')
        for link in path:
            link_cap = topo.get_link_capacity(link)
            num_flows_on_link = len(link_flows[link])
            fair_share = link_cap / num_flows_on_link
            min_share = min(min_share, fair_share)
        flow_throughput.append(min_share)

    link_utilizations = []
    for link, using_flows in link_flows.items():
        link_cap = topo.get_link_capacity(link)
        total_traffic = sum(flow_throughput[f] for f in using_flows)
        utilization = min(total_traffic / link_cap, 1.0) if link_cap > 0 else 0
        link_utilizations.append(utilization)

    if flow_throughput:
        avg_throughput = sum(flow_throughput) / len(flow_throughput)
        sorted_tp = sorted(flow_throughput)
        p5_idx = max(0, int(len(sorted_tp) * 0.05) - 1)
        p50_idx = len(sorted_tp) // 2
        p95_idx = min(len(sorted_tp) - 1, int(len(sorted_tp) * 0.95))

        p5_throughput = sorted_tp[p5_idx]
        p50_throughput = sorted_tp[p50_idx]
        p95_throughput = sorted_tp[p95_idx]
        min_throughput = min(flow_throughput)
        max_throughput = max(flow_throughput)
    else:
        avg_throughput = p5_throughput = p50_throughput = p95_throughput = 0
        min_throughput = max_throughput = 0

    if link_utilizations:
        avg_link_util = sum(link_utilizations) / len(link_utilizations)
        max_link_util = max(link_utilizations)
        num_bottleneck = sum(1 for u in link_utilizations if u > 0.8)
    else:
        avg_link_util = max_link_util = 0
        num_bottleneck = 0

    return OversubscriptionResult(
        k=k,
        oversubscription_ratio=oversubscription_ratio,
        num_flows=num_flows,
        link_capacity=1.0,
        average_throughput=avg_throughput,
        p5_throughput=p5_throughput,
        p50_throughput=p50_throughput,
        p95_throughput=p95_throughput,
        min_throughput=min_throughput,
        max_throughput=max_throughput,
        link_utilizations=link_utilizations,
        avg_link_utilization=avg_link_util,
        max_link_utilization=max_link_util,
        num_bottleneck_links=num_bottleneck,
        throughput_per_flow=flow_throughput,
    )


def run_oversubscription_experiment(
    k: int = 8,
    ratios: Optional[List[float]] = None,
    num_flows: int = 200,
    traffic_pattern: str = "random",
    seed: int = 42,
) -> List[OversubscriptionResult]:
    if ratios is None:
        ratios = [1.0, 2.0, 4.0, 8.0]

    results = []
    for ratio in ratios:
        result = simulate_oversubscription(
            k=k,
            oversubscription_ratio=ratio,
            num_flows=num_flows,
            traffic_pattern=traffic_pattern,
            seed=seed,
        )
        results.append(result)

    return results


def print_oversubscription_comparison(results: List[OversubscriptionResult]) -> str:
    if not results:
        return "No results to compare."

    lines = [
        "=" * 80,
        "OVERSUBSCRIPTION SIMULATION COMPARISON",
        "=" * 80,
        f"Fat-tree k={results[0].k}, {results[0].num_flows} flows",
        "-" * 80,
        f"{'Ratio':>8} | {'Avg Tput':>10} | {'P5 Tput':>10} | {'P50 Tput':>10} | {'Avg Link Util':>14} | {'Bottlenecks':>11}",
        "-" * 80,
    ]

    baseline_throughput = results[0].average_throughput if results else 1.0

    for r in results:
        lines.append(
            f"{r.oversubscription_ratio:>7.1f}:1 | "
            f"{r.average_throughput:>10.4f} | "
            f"{r.p5_throughput:>10.4f} | "
            f"{r.p50_throughput:>10.4f} | "
            f"{r.avg_link_utilization*100:>13.1f}% | "
            f"{r.num_bottleneck_links:>11}"
        )

    lines.append("-" * 80)

    if len(results) >= 2:
        ratio_1 = results[0]
        ratio_4_idx = next((i for i, r in enumerate(results) if r.oversubscription_ratio >= 4.0), -1)

        if ratio_4_idx >= 0:
            ratio_4 = results[ratio_4_idx]
            throughput_reduction = (ratio_1.average_throughput - ratio_4.average_throughput) / ratio_1.average_throughput * 100
            p5_reduction = (ratio_1.p5_throughput - ratio_4.p5_throughput) / ratio_1.p5_throughput * 100 if ratio_1.p5_throughput > 0 else 0

            lines.append("")
            lines.append("KEY INSIGHTS:")
            lines.append(f"  • Moving from 1:1 to 4:1 oversubscription reduces average throughput by {throughput_reduction:.1f}%")
            lines.append(f"  • 5th percentile (worst-case) throughput reduced by {p5_reduction:.1f}%")
            lines.append(f"  • At 4:1 ratio, {ratio_4.num_bottleneck_links} links experience >80% utilization")

            if ratio_4.p5_throughput < 0.5 * ratio_4.average_throughput:
                lines.append(f"  • Long-tail latency detected: P5 throughput is {ratio_4.p5_throughput/ratio_4.average_throughput*100:.0f}% of average")

    lines.append("=" * 80)
    return "\n".join(lines)
