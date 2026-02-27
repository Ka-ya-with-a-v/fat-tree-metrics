from typing import List, Optional, Tuple

from models import FatTreeMetrics, FatTreeValidationError


def validate_k(k: int) -> None:
    if not isinstance(k, int):
        raise FatTreeValidationError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise FatTreeValidationError(f"k must be positive, got {k}")
    if k % 2 != 0:
        raise FatTreeValidationError(f"k must be even for valid fat-tree, got {k}")


def compute_fat_tree_metrics(k: int, link_capacity_gbps: float = 10.0) -> FatTreeMetrics:
    validate_k(k)

    half_k = k // 2

    pods = k
    core_switches = half_k ** 2
    aggregation_switches = k * half_k
    edge_switches = k * half_k
    total_switches = core_switches + aggregation_switches + edge_switches

    total_hosts = (k ** 3) // 4

    host_to_edge_links = total_hosts
    edge_to_agg_links = k * (half_k ** 2)
    agg_to_core_links = k * (half_k ** 2)
    total_links = host_to_edge_links + edge_to_agg_links + agg_to_core_links

    bisection_links = (k ** 3) // 8
    bisection_bandwidth = bisection_links * link_capacity_gbps

    interpod_paths = half_k ** 2
    max_hops = 4
    oversubscription = 1.0

    return FatTreeMetrics(
        k=k,
        link_capacity_gbps=link_capacity_gbps,
        hosts=total_hosts,
        pods=pods,
        core_switches=core_switches,
        aggregation_switches=aggregation_switches,
        edge_switches=edge_switches,
        total_switches=total_switches,
        total_links=total_links,
        bisection_links=bisection_links,
        bisection_bandwidth_gbps=bisection_bandwidth,
        interpod_paths=interpod_paths,
        max_hops=max_hops,
        oversubscription_ratio=oversubscription,
    )


def compare_topologies(metrics_a: FatTreeMetrics, metrics_b: FatTreeMetrics) -> dict:
    return {
        "k_ratio": metrics_b.k / metrics_a.k,
        "host_ratio": metrics_b.hosts / metrics_a.hosts,
        "switch_ratio": metrics_b.total_switches / metrics_a.total_switches,
        "bandwidth_ratio": metrics_b.bisection_bandwidth_gbps / metrics_a.bisection_bandwidth_gbps,
        "host_increase": metrics_b.hosts - metrics_a.hosts,
        "switch_increase": metrics_b.total_switches - metrics_a.total_switches,
    }
