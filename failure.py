from models import FailureAnalysis, FatTreeValidationError
from metrics import validate_k


def analyze_failure_impact(k: int) -> FailureAnalysis:
    validate_k(k)

    half_k = k // 2
    total_hosts = (k ** 3) // 4
    core_switches = half_k ** 2

    total_host_pairs = (total_hosts * (total_hosts - 1)) // 2

    interpod_hosts_per_pod = total_hosts // k
    interpod_pairs = k * (k - 1) // 2 * (interpod_hosts_per_pod ** 2)
    core_failure_impact_pct = (1 / core_switches) * (interpod_pairs / total_host_pairs) * 100

    hosts_per_agg = (k // 2) * (k // 2)
    agg_affected_pairs = hosts_per_agg * (total_hosts - hosts_per_agg)
    agg_failure_impact_pct = (0.5 / half_k) * (agg_affected_pairs / total_host_pairs) * 100

    hosts_per_edge = half_k

    intrapod_paths = half_k
    interpod_paths = half_k ** 2

    critical = ["Edge switches (single point of failure for connected hosts)"]
    if k <= 4:
        critical.append("Core switches (limited redundancy with small k)")

    if k >= 24:
        rating = "EXCELLENT - High path diversity"
    elif k >= 8:
        rating = "GOOD - Adequate redundancy"
    else:
        rating = "MODERATE - Limited path diversity"

    return FailureAnalysis(
        k=k,
        single_core_failure_host_pairs_affected_pct=core_failure_impact_pct,
        single_agg_failure_host_pairs_affected_pct=agg_failure_impact_pct,
        single_edge_failure_hosts_disconnected=hosts_per_edge,
        single_link_redundancy="No disconnection (alternate paths exist)",
        intrapod_paths=intrapod_paths,
        interpod_paths=interpod_paths,
        critical_components=critical,
        resilience_rating=rating,
    )


def simulate_cascading_failures(
    k: int,
    failed_core_switches: int = 0,
    failed_agg_switches: int = 0,
    failed_edge_switches: int = 0,
) -> dict:
    validate_k(k)

    half_k = k // 2

    orig_core = half_k ** 2
    orig_agg = k * half_k
    orig_edge = k * half_k
    orig_hosts = (k ** 3) // 4

    rem_core = max(0, orig_core - failed_core_switches)
    rem_agg = max(0, orig_agg - failed_agg_switches)
    rem_edge = max(0, orig_edge - failed_edge_switches)

    hosts_disconnected = failed_edge_switches * half_k
    connected_hosts = orig_hosts - hosts_disconnected

    core_capacity_pct = (rem_core / orig_core * 100) if orig_core > 0 else 0
    agg_capacity_pct = (rem_agg / orig_agg * 100) if orig_agg > 0 else 0

    interpod_connected = rem_core > 0

    orig_bisection_paths = half_k ** 2
    remaining_bisection_pct = (rem_core / orig_bisection_paths * 100) if orig_bisection_paths > 0 else 0

    return {
        "original_hosts": orig_hosts,
        "connected_hosts": connected_hosts,
        "hosts_disconnected": hosts_disconnected,
        "connectivity_pct": (connected_hosts / orig_hosts * 100) if orig_hosts > 0 else 0,
        "core_capacity_remaining_pct": core_capacity_pct,
        "aggregation_capacity_remaining_pct": agg_capacity_pct,
        "interpod_connectivity": interpod_connected,
        "bisection_bandwidth_remaining_pct": remaining_bisection_pct,
        "status": "CRITICAL" if not interpod_connected else ("DEGRADED" if remaining_bisection_pct < 50 else "OPERATIONAL"),
    }
