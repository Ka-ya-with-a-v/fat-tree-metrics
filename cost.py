from typing import List, Optional, Tuple

from models import FatTreeMetrics, CostEstimate, SwitchPricing
from metrics import compute_fat_tree_metrics


def estimate_network_cost(
    metrics: FatTreeMetrics,
    pricing: Optional[SwitchPricing] = None
) -> CostEstimate:
    if pricing is None:
        pricing = SwitchPricing()

    core_cost = metrics.core_switches * pricing.core_switch_price
    agg_cost = metrics.aggregation_switches * pricing.aggregation_switch_price
    edge_cost = metrics.edge_switches * pricing.edge_switch_price
    total_switch_cost = core_cost + agg_cost + edge_cost

    total_link_bandwidth = metrics.total_links * metrics.link_capacity_gbps
    link_cost = total_link_bandwidth * pricing.link_cost_per_gbps

    total_cost = total_switch_cost + link_cost
    cost_per_host = total_cost / metrics.hosts if metrics.hosts > 0 else 0

    return CostEstimate(
        core_switch_cost=core_cost,
        aggregation_switch_cost=agg_cost,
        edge_switch_cost=edge_cost,
        total_switch_cost=total_switch_cost,
        link_cost=link_cost,
        total_cost=total_cost,
        cost_per_host=cost_per_host,
    )


def compare_cost_scaling(
    k_values: List[int],
    pricing: Optional[SwitchPricing] = None
) -> List[Tuple[int, CostEstimate]]:
    results = []
    for k in k_values:
        metrics = compute_fat_tree_metrics(k)
        cost = estimate_network_cost(metrics, pricing)
        results.append((k, cost))
    return results
