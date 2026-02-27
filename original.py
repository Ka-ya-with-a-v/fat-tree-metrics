"""
Fat-Tree Network Topology Analyzer

A production-quality module for computing and analyzing k-ary fat-tree
data center network metrics. Supports scaling experiments, comparisons,
visualization, cost estimation, failure analysis, and oversubscription simulation.

Author: Kavya
Date: 2026-02-27
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from enum import Enum
import json
import csv
import random
from pathlib import Path
from collections import defaultdict

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not installed. Visualization disabled. Install with: pip install matplotlib")

# Optional: numpy for statistical analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠ numpy not installed. Some statistics disabled. Install with: pip install numpy")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class FatTreeSize(Enum):
    """Standard fat-tree configurations used in industry/research."""
    SMALL = 4       # 16 hosts - lab/testing
    MEDIUM = 8      # 128 hosts - small cluster
    LARGE = 24      # 3,456 hosts - medium DC
    XLARGE = 48     # 27,648 hosts - large DC


@dataclass
class SwitchPricing:
    """
    Network switch pricing model.
    
    Prices are approximate and vary by vendor/features.
    Default values based on typical enterprise pricing (2026).
    """
    core_switch_price: float = 50_000.0      # High-capacity core switches
    aggregation_switch_price: float = 25_000.0  # Mid-tier aggregation
    edge_switch_price: float = 5_000.0       # ToR/edge switches
    link_cost_per_gbps: float = 100.0        # Per-Gbps cabling/optics cost
    
    def __str__(self) -> str:
        return (
            f"Switch Pricing:\n"
            f"  Core: ${self.core_switch_price:,.0f}\n"
            f"  Aggregation: ${self.aggregation_switch_price:,.0f}\n"
            f"  Edge: ${self.edge_switch_price:,.0f}\n"
            f"  Link cost/Gbps: ${self.link_cost_per_gbps:,.0f}"
        )


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True, slots=True)
class FatTreeMetrics:
    """
    Immutable container for k-ary fat-tree network metrics.
    
    A fat-tree is a special case of a Clos network topology commonly used
    in data centers for its full bisection bandwidth and path diversity.
    
    Attributes:
        k: Number of ports per switch (must be even)
        link_capacity_gbps: Capacity of each link in Gbps
        hosts: Total number of end hosts
        pods: Number of pods in the topology
        core_switches: Number of core-layer switches
        aggregation_switches: Number of aggregation-layer switches
        edge_switches: Number of edge/ToR switches
        total_switches: Total switch count across all layers
        total_links: Total number of links in the network
        bisection_links: Number of links crossing the bisection
        bisection_bandwidth_gbps: Total bisection bandwidth
        interpod_paths: Number of equal-cost paths between pods
        max_hops: Maximum hop count between any two hosts
        oversubscription_ratio: Ratio of downlink to uplink capacity
    """
    k: int
    link_capacity_gbps: float
    hosts: int
    pods: int
    core_switches: int
    aggregation_switches: int
    edge_switches: int
    total_switches: int
    total_links: int
    bisection_links: int
    bisection_bandwidth_gbps: float
    interpod_paths: int
    max_hops: int
    oversubscription_ratio: float
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Fat-Tree (k={self.k})\n"
            f"  Hosts: {self.hosts:,}\n"
            f"  Pods: {self.pods}\n"
            f"  Switches: {self.total_switches:,} "
            f"(Core: {self.core_switches}, Agg: {self.aggregation_switches}, Edge: {self.edge_switches})\n"
            f"  Total Links: {self.total_links:,}\n"
            f"  Bisection BW: {self.bisection_bandwidth_gbps:,.0f} Gbps\n"
            f"  Inter-pod Paths: {self.interpod_paths}\n"
            f"  Max Hops: {self.max_hops}\n"
            f"  Oversubscription: {self.oversubscription_ratio}:1"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CostEstimate:
    """
    Network infrastructure cost breakdown.
    """
    core_switch_cost: float
    aggregation_switch_cost: float
    edge_switch_cost: float
    total_switch_cost: float
    link_cost: float
    total_cost: float
    cost_per_host: float
    
    def __str__(self) -> str:
        return (
            f"Cost Estimate:\n"
            f"  Core Switches:        ${self.core_switch_cost:>12,.0f}\n"
            f"  Aggregation Switches: ${self.aggregation_switch_cost:>12,.0f}\n"
            f"  Edge Switches:        ${self.edge_switch_cost:>12,.0f}\n"
            f"  ─────────────────────────────────────\n"
            f"  Total Switch Cost:    ${self.total_switch_cost:>12,.0f}\n"
            f"  Link/Cabling Cost:    ${self.link_cost:>12,.0f}\n"
            f"  ═══════════════════════════════════════\n"
            f"  TOTAL COST:           ${self.total_cost:>12,.0f}\n"
            f"  Cost per Host:        ${self.cost_per_host:>12,.2f}"
        )


@dataclass
class FailureAnalysis:
    """
    Analysis of network resilience under component failures.
    """
    k: int
    
    # Single failure impacts
    single_core_failure_host_pairs_affected_pct: float
    single_agg_failure_host_pairs_affected_pct: float
    single_edge_failure_hosts_disconnected: int
    single_link_redundancy: str
    
    # Path diversity
    intrapod_paths: int
    interpod_paths: int
    
    # Recommendations
    critical_components: List[str]
    resilience_rating: str
    
    def __str__(self) -> str:
        return (
            f"Failure Analysis (k={self.k}):\n"
            f"  ┌─────────────────────────────────────────────────────────┐\n"
            f"  │ SINGLE COMPONENT FAILURE IMPACT                         │\n"
            f"  ├─────────────────────────────────────────────────────────┤\n"
            f"  │ Core switch failure:    {self.single_core_failure_host_pairs_affected_pct:>5.2f}% host pairs affected  │\n"
            f"  │ Agg switch failure:     {self.single_agg_failure_host_pairs_affected_pct:>5.2f}% host pairs affected  │\n"
            f"  │ Edge switch failure:    {self.single_edge_failure_hosts_disconnected:>5} hosts disconnected    │\n"
            f"  │ Single link failure:    {self.single_link_redundancy:<28}│\n"
            f"  ├─────────────────────────────────────────────────────────┤\n"
            f"  │ PATH DIVERSITY                                          │\n"
            f"  ├─────────────────────────────────────────────────────────┤\n"
            f"  │ Intra-pod paths:        {self.intrapod_paths:>5}                           │\n"
            f"  │ Inter-pod paths:        {self.interpod_paths:>5}                           │\n"
            f"  ├─────────────────────────────────────────────────────────┤\n"
            f"  │ Resilience Rating:      {self.resilience_rating:<28}│\n"
            f"  └─────────────────────────────────────────────────────────┘\n"
            f"  Critical Components: {', '.join(self.critical_components)}"
        )


@dataclass
class OversubscriptionResult:
    """
    Results from oversubscription simulation.
    
    Attributes:
        k: Fat-tree parameter
        oversubscription_ratio: The simulated oversubscription ratio (e.g., 1.0, 2.0, 4.0)
        num_flows: Number of flows simulated
        link_capacity: Capacity of each link (normalized)
        average_throughput: Average per-flow throughput
        p5_throughput: 5th percentile throughput (worst 5% of flows)
        p50_throughput: Median throughput
        p95_throughput: 95th percentile throughput
        min_throughput: Minimum flow throughput
        max_throughput: Maximum flow throughput
        link_utilizations: List of link utilization values
        avg_link_utilization: Average link utilization
        max_link_utilization: Maximum link utilization
        num_bottleneck_links: Number of links at >80% utilization
        throughput_per_flow: List of individual flow throughputs
    """
    k: int
    oversubscription_ratio: float
    num_flows: int
    link_capacity: float
    average_throughput: float
    p5_throughput: float
    p50_throughput: float
    p95_throughput: float
    min_throughput: float
    max_throughput: float
    link_utilizations: List[float]
    avg_link_utilization: float
    max_link_utilization: float
    num_bottleneck_links: int
    throughput_per_flow: List[float]
    
    def __str__(self) -> str:
        return (
            f"Oversubscription Simulation Results (k={self.k}, ratio={self.oversubscription_ratio}:1)\n"
            f"  ┌────────────────────────────────────────────────────────────┐\n"
            f"  │ TRAFFIC MODEL                                              │\n"
            f"  ├────────────────────────────────────────────────────────────┤\n"
            f"  │ Number of flows:        {self.num_flows:>6}                            │\n"
            f"  │ Link capacity:          {self.link_capacity:>6.2f} (normalized)             │\n"
            f"  ├────────────────────────────────────────────────────────────┤\n"
            f"  │ THROUGHPUT STATISTICS (per flow)                           │\n"
            f"  ├────────────────────────────────────────────────────────────┤\n"
            f"  │ Average throughput:     {self.average_throughput:>6.4f}                          │\n"
            f"  │ Median throughput:      {self.p50_throughput:>6.4f}                          │\n"
            f"  │ 5th percentile:         {self.p5_throughput:>6.4f}                          │\n"
            f"  │ 95th percentile:        {self.p95_throughput:>6.4f}                          │\n"
            f"  │ Min/Max:                {self.min_throughput:>6.4f} / {self.max_throughput:<6.4f}              │\n"
            f"  ├────────────────────────────────────────────────────────────┤\n"
            f"  │ LINK UTILIZATION                                           │\n"
            f"  ├────────────────────────────────────────────────────────────┤\n"
            f"  │ Average utilization:    {self.avg_link_utilization*100:>6.2f}%                         │\n"
            f"  │ Maximum utilization:    {self.max_link_utilization*100:>6.2f}%                         │\n"
            f"  │ Bottleneck links (>80%):{self.num_bottleneck_links:>6}                            │\n"
            f"  └────────────────────────────────────────────────────────────┘"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding large lists for readability)."""
        return {
            "k": self.k,
            "oversubscription_ratio": self.oversubscription_ratio,
            "num_flows": self.num_flows,
            "link_capacity": self.link_capacity,
            "average_throughput": self.average_throughput,
            "p5_throughput": self.p5_throughput,
            "p50_throughput": self.p50_throughput,
            "p95_throughput": self.p95_throughput,
            "min_throughput": self.min_throughput,
            "max_throughput": self.max_throughput,
            "avg_link_utilization": self.avg_link_utilization,
            "max_link_utilization": self.max_link_utilization,
            "num_bottleneck_links": self.num_bottleneck_links,
        }


# ============================================================================
# EXCEPTIONS
# ============================================================================

class FatTreeValidationError(Exception):
    """Raised when fat-tree parameters are invalid."""
    pass


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def validate_k(k: int) -> None:
    """
    Validate that k is a valid fat-tree parameter.
    
    Args:
        k: Number of ports per switch
        
    Raises:
        FatTreeValidationError: If k is not a positive even integer
    """
    if not isinstance(k, int):
        raise FatTreeValidationError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise FatTreeValidationError(f"k must be positive, got {k}")
    if k % 2 != 0:
        raise FatTreeValidationError(f"k must be even for valid fat-tree, got {k}")


def compute_fat_tree_metrics(
    k: int,
    link_capacity_gbps: float = 10.0
) -> FatTreeMetrics:
    """
    Compute all structural and bandwidth metrics for a k-ary fat-tree.
    
    In a k-ary fat-tree:
    - Each pod contains k switches: k/2 edge + k/2 aggregation
    - There are k pods total
    - (k/2)² core switches connect pods
    - Each edge switch connects to k/2 hosts
    
    Args:
        k: Number of ports per switch (must be even positive integer)
        link_capacity_gbps: Link bandwidth in Gbps (default: 10)
        
    Returns:
        FatTreeMetrics dataclass with all computed values
        
    Raises:
        FatTreeValidationError: If k is invalid
        
    Example:
        >>> metrics = compute_fat_tree_metrics(4)
        >>> metrics.hosts
        16
        >>> metrics.total_switches
        20
    """
    validate_k(k)
    
    half_k = k // 2
    
    # Structural metrics
    pods = k
    core_switches = half_k ** 2
    aggregation_switches = k * half_k  # k/2 per pod × k pods
    edge_switches = k * half_k          # k/2 per pod × k pods
    total_switches = core_switches + aggregation_switches + edge_switches
    
    # Host count: k/2 hosts per edge switch × k/2 edge switches per pod × k pods
    total_hosts = (k ** 3) // 4
    
    # Link count:
    # - Host to edge: total_hosts links
    # - Edge to aggregation: (k/2) * (k/2) per pod * k pods = k * (k/2)^2
    # - Aggregation to core: k * (k/2) * (k/2) = same formula
    host_to_edge_links = total_hosts
    edge_to_agg_links = k * (half_k ** 2)
    agg_to_core_links = k * (half_k ** 2)
    total_links = host_to_edge_links + edge_to_agg_links + agg_to_core_links
    
    # Bisection analysis
    bisection_links = (k ** 3) // 8
    bisection_bandwidth = bisection_links * link_capacity_gbps
    
    # Path metrics
    interpod_paths = half_k ** 2  # Equal-cost paths via core
    max_hops = 4  # Always 4 hops in fat-tree: edge→agg→core→agg→edge
    
    # Oversubscription: 1:1 in ideal fat-tree (full bisection bandwidth)
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


# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_network_cost(
    metrics: FatTreeMetrics,
    pricing: Optional[SwitchPricing] = None
) -> CostEstimate:
    """
    Estimate total network infrastructure cost.
    
    Args:
        metrics: Computed fat-tree metrics
        pricing: Switch pricing model (uses defaults if not provided)
        
    Returns:
        CostEstimate with detailed cost breakdown
        
    Example:
        >>> metrics = compute_fat_tree_metrics(8)
        >>> cost = estimate_network_cost(metrics)
        >>> print(f"Total: ${cost.total_cost:,.0f}")
    """
    if pricing is None:
        pricing = SwitchPricing()
    
    # Switch costs
    core_cost = metrics.core_switches * pricing.core_switch_price
    agg_cost = metrics.aggregation_switches * pricing.aggregation_switch_price
    edge_cost = metrics.edge_switches * pricing.edge_switch_price
    total_switch_cost = core_cost + agg_cost + edge_cost
    
    # Link costs (total bandwidth * cost per Gbps)
    # Each link is link_capacity_gbps, we have total_links
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
    """
    Compare costs across different fat-tree sizes.
    
    Args:
        k_values: List of k values to analyze
        pricing: Switch pricing model
        
    Returns:
        List of (k, CostEstimate) tuples
    """
    results = []
    for k in k_values:
        metrics = compute_fat_tree_metrics(k)
        cost = estimate_network_cost(metrics, pricing)
        results.append((k, cost))
    return results


# ============================================================================
# FAILURE ANALYSIS
# ============================================================================

def analyze_failure_impact(k: int) -> FailureAnalysis:
    """
    Analyze network resilience and failure impact for a k-ary fat-tree.
    
    This analyzes:
    - Single component failure impact
    - Path diversity and redundancy
    - Critical components identification
    
    Args:
        k: Fat-tree parameter
        
    Returns:
        FailureAnalysis with detailed resilience metrics
        
    Note:
        Fat-trees are designed for high resilience. Key properties:
        - No single link failure disconnects any host pair
        - Multiple equal-cost paths provide redundancy
        - Edge switches are single points of failure for their hosts
    """
    validate_k(k)
    
    half_k = k // 2
    total_hosts = (k ** 3) // 4
    core_switches = half_k ** 2
    
    # Calculate total host pairs for percentage calculations
    total_host_pairs = (total_hosts * (total_hosts - 1)) // 2
    
    # === Single Core Switch Failure ===
    # Each core switch handles 1/(k/2)² of inter-pod traffic
    # With (k/2)² paths between pods, losing one reduces capacity by 1/(k/2)²
    # But no host pairs are disconnected (other paths available)
    # Affected percentage = fraction of bandwidth lost for inter-pod pairs
    interpod_hosts_per_pod = total_hosts // k
    interpod_pairs = k * (k - 1) // 2 * (interpod_hosts_per_pod ** 2)
    core_failure_impact_pct = (1 / core_switches) * (interpod_pairs / total_host_pairs) * 100
    
    # === Single Aggregation Switch Failure ===
    # Each agg switch connects half of pod's edge switches to half of core switches
    # Losing one reduces intra-pod and inter-pod capacity for affected hosts
    hosts_per_agg = (k // 2) * (k // 2)  # hosts connected via this agg
    agg_affected_pairs = hosts_per_agg * (total_hosts - hosts_per_agg)
    agg_failure_impact_pct = (0.5 / half_k) * (agg_affected_pairs / total_host_pairs) * 100
    
    # === Single Edge Switch Failure ===
    # Edge switches are single points of failure - their hosts are disconnected
    hosts_per_edge = half_k
    
    # === Path Diversity ===
    # Intra-pod: paths through different aggregation switches
    intrapod_paths = half_k  # Can go through any of k/2 agg switches
    interpod_paths = half_k ** 2  # Can go through any of (k/2)² core switches
    
    # === Critical Components ===
    critical = ["Edge switches (single point of failure for connected hosts)"]
    if k <= 4:
        critical.append("Core switches (limited redundancy with small k)")
    
    # === Resilience Rating ===
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
    """
    Simulate impact of multiple simultaneous failures.
    
    Args:
        k: Fat-tree parameter
        failed_core_switches: Number of failed core switches
        failed_agg_switches: Number of failed aggregation switches  
        failed_edge_switches: Number of failed edge switches
        
    Returns:
        Dictionary with remaining capacity and connectivity metrics
    """
    validate_k(k)
    
    half_k = k // 2
    
    # Original counts
    orig_core = half_k ** 2
    orig_agg = k * half_k
    orig_edge = k * half_k
    orig_hosts = (k ** 3) // 4
    
    # Remaining after failures
    rem_core = max(0, orig_core - failed_core_switches)
    rem_agg = max(0, orig_agg - failed_agg_switches)
    rem_edge = max(0, orig_edge - failed_edge_switches)
    
    # Connectivity impact
    hosts_disconnected = failed_edge_switches * half_k
    connected_hosts = orig_hosts - hosts_disconnected
    
    # Capacity impact
    core_capacity_pct = (rem_core / orig_core * 100) if orig_core > 0 else 0
    agg_capacity_pct = (rem_agg / orig_agg * 100) if orig_agg > 0 else 0
    
    # Inter-pod connectivity (requires at least 1 core switch)
    interpod_connected = rem_core > 0
    
    # Bisection bandwidth remaining
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


# ============================================================================
# OVERSUBSCRIPTION SIMULATION
# ============================================================================

class FatTreeTopology:
    """
    Fat-tree topology representation for traffic simulation.
    
    This class builds an abstract model of a fat-tree topology to simulate
    traffic flows and measure the impact of oversubscription.
    
    The topology consists of:
    - Hosts assigned to edge switches
    - Edge switches connected to aggregation switches
    - Aggregation switches connected to core switches
    - ECMP routing between any host pair
    """
    
    def __init__(self, k: int, oversubscription_ratio: float = 1.0):
        """
        Initialize fat-tree topology.
        
        Args:
            k: Number of ports per switch (must be even)
            oversubscription_ratio: Ratio of downlink to uplink capacity.
                                    1.0 = full bisection bandwidth
                                    2.0 = 2:1 oversubscription (half uplinks)
                                    4.0 = 4:1 oversubscription (quarter uplinks)
        """
        validate_k(k)
        self.k = k
        self.half_k = k // 2
        self.oversubscription_ratio = oversubscription_ratio
        
        # Compute topology parameters
        self.num_pods = k
        self.num_hosts = (k ** 3) // 4
        self.hosts_per_edge = self.half_k
        self.edge_per_pod = self.half_k
        self.agg_per_pod = self.half_k
        self.num_core = self.half_k ** 2
        
        # Build host-to-location mapping
        # Host ID -> (pod, edge_switch_in_pod, host_index_in_edge)
        self.host_location = {}
        host_id = 0
        for pod in range(self.num_pods):
            for edge in range(self.edge_per_pod):
                for h in range(self.hosts_per_edge):
                    self.host_location[host_id] = (pod, edge, h)
                    host_id += 1
        
        # Compute effective uplink capacity based on oversubscription
        # In full fat-tree: each edge has k/2 downlinks and k/2 uplinks (1:1)
        # With oversubscription r:1, effective uplinks = (k/2) / r
        self.effective_uplinks_per_edge = self.half_k / oversubscription_ratio
        self.effective_uplinks_per_agg = self.half_k / oversubscription_ratio
    
    def get_host_pod(self, host_id: int) -> int:
        """Get the pod number for a host."""
        return self.host_location[host_id][0]
    
    def get_host_edge(self, host_id: int) -> Tuple[int, int]:
        """Get (pod, edge_switch_index) for a host."""
        loc = self.host_location[host_id]
        return (loc[0], loc[1])
    
    def get_path_links(self, src_host: int, dst_host: int) -> List[str]:
        """
        Get the list of link identifiers traversed by a flow.
        
        For fat-tree with ECMP, we select one random path among equal-cost paths.
        
        Link naming:
        - "host_{h}_edge_{p}_{e}": Host h to edge switch e in pod p
        - "edge_{p}_{e}_agg_{p}_{a}": Edge e to agg a in pod p
        - "agg_{p}_{a}_core_{c}": Agg a in pod p to core c
        
        Returns:
            List of link identifiers (strings)
        """
        src_pod, src_edge, _ = self.host_location[src_host]
        dst_pod, dst_edge, _ = self.host_location[dst_host]
        
        links = []
        
        # Source host to source edge
        links.append(f"host_{src_host}_edge_{src_pod}_{src_edge}")
        
        if src_pod == dst_pod and src_edge == dst_edge:
            # Same edge switch - only host-edge links
            links.append(f"edge_{dst_pod}_{dst_edge}_host_{dst_host}")
            return links
        
        # Choose random aggregation switch in source pod
        src_agg = random.randint(0, self.agg_per_pod - 1)
        links.append(f"edge_{src_pod}_{src_edge}_agg_{src_pod}_{src_agg}")
        
        if src_pod == dst_pod:
            # Same pod, different edge - go through agg only
            links.append(f"agg_{dst_pod}_{src_agg}_edge_{dst_pod}_{dst_edge}")
        else:
            # Different pods - go through core
            # Choose random core switch (connected to this agg)
            # Core switches are indexed based on agg position
            core_base = src_agg * self.half_k
            core_offset = random.randint(0, self.half_k - 1)
            core_id = core_base + core_offset
            
            links.append(f"agg_{src_pod}_{src_agg}_core_{core_id}")
            
            # Core to destination aggregation
            dst_agg = core_offset  # The agg in dst pod connected to this core
            links.append(f"core_{core_id}_agg_{dst_pod}_{dst_agg}")
            
            # Agg to edge in destination pod
            links.append(f"agg_{dst_pod}_{dst_agg}_edge_{dst_pod}_{dst_edge}")
        
        # Destination edge to destination host
        links.append(f"edge_{dst_pod}_{dst_edge}_host_{dst_host}")
        
        return links
    
    def get_link_capacity(self, link_id: str) -> float:
        """
        Get the capacity of a link considering oversubscription.
        
        Uplinks (edge-agg, agg-core) have reduced capacity with oversubscription.
        
        Args:
            link_id: Link identifier string
            
        Returns:
            Link capacity (normalized, 1.0 = full capacity)
        """
        if "host_" in link_id and "edge_" in link_id:
            # Host-edge links always have full capacity
            return 1.0
        elif "edge_" in link_id and "agg_" in link_id:
            # Edge-agg uplinks: affected by oversubscription
            return 1.0 / self.oversubscription_ratio
        elif "agg_" in link_id and "core_" in link_id:
            # Agg-core uplinks: affected by oversubscription
            return 1.0 / self.oversubscription_ratio
        elif "core_" in link_id:
            # Core links: affected by oversubscription
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
    """
    Simulate traffic flows under different oversubscription ratios.
    
    This function:
    1. Creates a fat-tree topology with the given oversubscription ratio
    2. Generates random host-to-host flows
    3. Routes flows using ECMP
    4. Computes fair-share throughput based on link contention
    5. Returns comprehensive statistics
    
    Traffic Patterns:
    - "random": Random source-destination pairs
    - "all_to_all": Subset of all-to-all traffic matrix
    - "intra_pod": Traffic within same pod only
    - "inter_pod": Traffic between different pods only
    
    Args:
        k: Fat-tree parameter
        oversubscription_ratio: Ratio of downlink to uplink capacity (1.0, 2.0, 4.0, 8.0)
        num_flows: Number of flows to simulate
        traffic_pattern: Traffic generation pattern
        seed: Random seed for reproducibility
        
    Returns:
        OversubscriptionResult with throughput and utilization statistics
        
    Example:
        >>> result = simulate_oversubscription(k=8, oversubscription_ratio=4.0, num_flows=200)
        >>> print(f"Avg throughput: {result.average_throughput:.4f}")
    """
    if seed is not None:
        random.seed(seed)
    
    # Build topology
    topo = FatTreeTopology(k, oversubscription_ratio)
    
    # Generate flows based on traffic pattern
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
        # Generate all possible pairs, then sample
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
    
    # Route flows and count link usage
    # link_id -> list of flow indices using this link
    link_flows: Dict[str, List[int]] = defaultdict(list)
    flow_paths: List[List[str]] = []
    
    for flow_idx, (src, dst) in enumerate(flows):
        path = topo.get_path_links(src, dst)
        flow_paths.append(path)
        for link in path:
            link_flows[link].append(flow_idx)
    
    # Compute fair-share throughput using max-min fairness
    # Each flow gets throughput = min over all links in path of (link_capacity / num_flows_on_link)
    flow_throughput = []
    
    for flow_idx, path in enumerate(flow_paths):
        # Find bottleneck: minimum fair share across all links in path
        min_share = float('inf')
        for link in path:
            link_cap = topo.get_link_capacity(link)
            num_flows_on_link = len(link_flows[link])
            fair_share = link_cap / num_flows_on_link
            min_share = min(min_share, fair_share)
        flow_throughput.append(min_share)
    
    # Compute link utilizations
    link_utilizations = []
    for link, using_flows in link_flows.items():
        link_cap = topo.get_link_capacity(link)
        # Total traffic on link = sum of throughputs of flows using it
        total_traffic = sum(flow_throughput[f] for f in using_flows)
        utilization = min(total_traffic / link_cap, 1.0) if link_cap > 0 else 0
        link_utilizations.append(utilization)
    
    # Compute statistics
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
    """
    Run oversubscription simulation across multiple ratios.
    
    Args:
        k: Fat-tree parameter
        ratios: List of oversubscription ratios to test (default: [1, 2, 4, 8])
        num_flows: Number of flows to simulate
        traffic_pattern: Traffic generation pattern
        seed: Random seed for reproducibility
        
    Returns:
        List of OversubscriptionResult for each ratio
    """
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
    """
    Generate a comparison table of oversubscription simulation results.
    
    Args:
        results: List of OversubscriptionResult objects
        
    Returns:
        Formatted ASCII table string
    """
    if not results:
        return "No results to compare."
    
    # Header
    lines = [
        "=" * 80,
        "OVERSUBSCRIPTION SIMULATION COMPARISON",
        "=" * 80,
        f"Fat-tree k={results[0].k}, {results[0].num_flows} flows",
        "-" * 80,
        f"{'Ratio':>8} | {'Avg Tput':>10} | {'P5 Tput':>10} | {'P50 Tput':>10} | {'Avg Link Util':>14} | {'Bottlenecks':>11}",
        "-" * 80,
    ]
    
    # Baseline for comparison
    baseline_throughput = results[0].average_throughput if results else 1.0
    
    for r in results:
        reduction_pct = ((baseline_throughput - r.average_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0
        lines.append(
            f"{r.oversubscription_ratio:>7.1f}:1 | "
            f"{r.average_throughput:>10.4f} | "
            f"{r.p5_throughput:>10.4f} | "
            f"{r.p50_throughput:>10.4f} | "
            f"{r.avg_link_utilization*100:>13.1f}% | "
            f"{r.num_bottleneck_links:>11}"
        )
    
    lines.append("-" * 80)
    
    # Summary insights
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
            
            # Check for long-tail behavior
            if ratio_4.p5_throughput < 0.5 * ratio_4.average_throughput:
                lines.append(f"  • Long-tail latency detected: P5 throughput is {ratio_4.p5_throughput/ratio_4.average_throughput*100:.0f}% of average")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cubic_scaling(
    k_values: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize the cubic scaling relationship in fat-tree topologies.
    
    Plots hosts, switches, and bisection bandwidth against k value,
    demonstrating the O(k³) scaling of hosts vs O(k²) scaling of switches.
    
    Args:
        k_values: List of k values to plot (default: 4, 6, 8, ..., 48)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot interactively
        
    Raises:
        ImportError: If matplotlib is not installed
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    if k_values is None:
        k_values = list(range(4, 50, 2))  # Even numbers from 4 to 48
    
    # Compute metrics for each k
    metrics_list = [compute_fat_tree_metrics(k) for k in k_values]
    
    hosts = [m.hosts for m in metrics_list]
    switches = [m.total_switches for m in metrics_list]
    bisection_bw = [m.bisection_bandwidth_gbps for m in metrics_list]
    core_switches = [m.core_switches for m in metrics_list]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fat-Tree Topology Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Hosts (cubic scaling)
    ax1 = axes[0, 0]
    ax1.plot(k_values, hosts, 'b-o', linewidth=2, markersize=6, label='Actual')
    ax1.plot(k_values, [(k**3)/4 for k in k_values], 'r--', alpha=0.7, label='k³/4 (theoretical)')
    ax1.set_xlabel('k (ports per switch)', fontsize=11)
    ax1.set_ylabel('Number of Hosts', fontsize=11)
    ax1.set_title('Host Count - O(k³) Scaling', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot 2: Switches (quadratic scaling)
    ax2 = axes[0, 1]
    ax2.plot(k_values, switches, 'g-s', linewidth=2, markersize=6, label='Total Switches')
    ax2.plot(k_values, core_switches, 'm-^', linewidth=2, markersize=6, label='Core Switches')
    ax2.plot(k_values, [(5*k**2)/4 for k in k_values], 'r--', alpha=0.7, label='5k²/4 (theoretical)')
    ax2.set_xlabel('k (ports per switch)', fontsize=11)
    ax2.set_ylabel('Number of Switches', fontsize=11)
    ax2.set_title('Switch Count - O(k²) Scaling', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot 3: Bisection Bandwidth
    ax3 = axes[1, 0]
    ax3.plot(k_values, bisection_bw, 'c-D', linewidth=2, markersize=6)
    ax3.fill_between(k_values, bisection_bw, alpha=0.3)
    ax3.set_xlabel('k (ports per switch)', fontsize=11)
    ax3.set_ylabel('Bisection Bandwidth (Gbps)', fontsize=11)
    ax3.set_title('Bisection Bandwidth - O(k³) Scaling', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot 4: Hosts per Switch (efficiency)
    ax4 = axes[1, 1]
    hosts_per_switch = [h/s for h, s in zip(hosts, switches)]
    ax4.bar(k_values, hosts_per_switch, color='orange', alpha=0.7, width=1.5)
    ax4.axhline(y=sum(hosts_per_switch)/len(hosts_per_switch), color='red', 
                linestyle='--', label=f'Average: {sum(hosts_per_switch)/len(hosts_per_switch):.1f}')
    ax4.set_xlabel('k (ports per switch)', fontsize=11)
    ax4.set_ylabel('Hosts per Switch', fontsize=11)
    ax4.set_title('Network Efficiency (Hosts/Switch)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_cost_analysis(
    k_values: Optional[List[int]] = None,
    pricing: Optional[SwitchPricing] = None,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize network cost breakdown and scaling.
    
    Args:
        k_values: List of k values to analyze
        pricing: Switch pricing model
        save_path: Path to save the figure
        show_plot: Whether to display interactively
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    if k_values is None:
        k_values = [4, 8, 16, 24, 32, 48]
    
    if pricing is None:
        pricing = SwitchPricing()
    
    # Compute costs
    costs = []
    for k in k_values:
        metrics = compute_fat_tree_metrics(k)
        cost = estimate_network_cost(metrics, pricing)
        costs.append(cost)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Fat-Tree Network Cost Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Total Cost
    ax1 = axes[0]
    total_costs = [c.total_cost / 1_000_000 for c in costs]  # In millions
    bars = ax1.bar(range(len(k_values)), total_costs, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(k_values)))
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.set_ylabel('Total Cost ($ Millions)', fontsize=11)
    ax1.set_title('Total Network Cost', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cost in zip(bars, total_costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${cost:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Cost Breakdown (Stacked)
    ax2 = axes[1]
    core_costs = [c.core_switch_cost / 1_000_000 for c in costs]
    agg_costs = [c.aggregation_switch_cost / 1_000_000 for c in costs]
    edge_costs = [c.edge_switch_cost / 1_000_000 for c in costs]
    link_costs = [c.link_cost / 1_000_000 for c in costs]
    
    x = range(len(k_values))
    ax2.bar(x, core_costs, label='Core Switches', color='#e74c3c', alpha=0.8)
    ax2.bar(x, agg_costs, bottom=core_costs, label='Aggregation', color='#f39c12', alpha=0.8)
    ax2.bar(x, edge_costs, bottom=[c+a for c,a in zip(core_costs, agg_costs)], 
            label='Edge Switches', color='#27ae60', alpha=0.8)
    ax2.bar(x, link_costs, bottom=[c+a+e for c,a,e in zip(core_costs, agg_costs, edge_costs)],
            label='Links/Cabling', color='#3498db', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.set_ylabel('Cost ($ Millions)', fontsize=11)
    ax2.set_title('Cost Breakdown by Component', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cost per Host
    ax3 = axes[2]
    cost_per_host = [c.cost_per_host for c in costs]
    ax3.plot(k_values, cost_per_host, 'g-o', linewidth=2, markersize=8)
    ax3.fill_between(k_values, cost_per_host, alpha=0.3, color='green')
    ax3.set_xlabel('k (ports per switch)', fontsize=11)
    ax3.set_ylabel('Cost per Host ($)', fontsize=11)
    ax3.set_title('Economy of Scale', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add annotations for first and last
    ax3.annotate(f'${cost_per_host[0]:,.0f}', (k_values[0], cost_per_host[0]),
                textcoords="offset points", xytext=(10, 10), fontsize=10)
    ax3.annotate(f'${cost_per_host[-1]:,.0f}', (k_values[-1], cost_per_host[-1]),
                textcoords="offset points", xytext=(-30, 10), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_failure_resilience(
    k: int = 24,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize network resilience under increasing failures.
    
    Args:
        k: Fat-tree parameter to analyze
        save_path: Path to save the figure
        show_plot: Whether to display interactively
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    validate_k(k)
    
    half_k = k // 2
    max_core = half_k ** 2
    max_agg = k * half_k
    max_edge = k * half_k
    
    # Simulate increasing failures
    core_failures = list(range(0, max_core + 1, max(1, max_core // 10)))
    edge_failures = list(range(0, min(max_edge + 1, 50), max(1, max_edge // 20)))
    
    # Core failure impact
    core_connectivity = []
    core_bisection = []
    for f in core_failures:
        result = simulate_cascading_failures(k, failed_core_switches=f)
        core_connectivity.append(result['connectivity_pct'])
        core_bisection.append(result['bisection_bandwidth_remaining_pct'])
    
    # Edge failure impact
    edge_connectivity = []
    for f in edge_failures:
        result = simulate_cascading_failures(k, failed_edge_switches=f)
        edge_connectivity.append(result['connectivity_pct'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Fat-Tree (k={k}) Resilience Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Core switch failures
    ax1 = axes[0]
    ax1.plot(core_failures, core_bisection, 'r-o', linewidth=2, label='Bisection BW')
    ax1.plot(core_failures, core_connectivity, 'b-s', linewidth=2, label='Host Connectivity')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.fill_between(core_failures, core_bisection, alpha=0.2, color='red')
    ax1.set_xlabel('Number of Failed Core Switches', fontsize=11)
    ax1.set_ylabel('Remaining Capacity (%)', fontsize=11)
    ax1.set_title('Impact of Core Switch Failures', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Plot 2: Edge switch failures
    ax2 = axes[1]
    ax2.plot(edge_failures, edge_connectivity, 'g-o', linewidth=2)
    ax2.fill_between(edge_failures, edge_connectivity, alpha=0.3, color='green')
    ax2.set_xlabel('Number of Failed Edge Switches', fontsize=11)
    ax2.set_ylabel('Host Connectivity (%)', fontsize=11)
    ax2.set_title('Impact of Edge Switch Failures', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    # Add annotation
    if len(edge_failures) > 5:
        mid_idx = len(edge_failures) // 2
        ax2.annotate(f'{edge_connectivity[mid_idx]:.0f}% connected\nwith {edge_failures[mid_idx]} failures',
                    (edge_failures[mid_idx], edge_connectivity[mid_idx]),
                    textcoords="offset points", xytext=(20, -20), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Resilience analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_oversubscription_analysis(
    results: List[OversubscriptionResult],
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize oversubscription simulation results.
    
    Creates two plots:
    1. Oversubscription ratio vs average throughput per flow
    2. CDF/histogram of link utilization for different ratios
    
    Args:
        results: List of OversubscriptionResult from simulation
        save_path: Path to save the figure
        show_plot: Whether to display interactively
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    if not results:
        print("No results to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Oversubscription Impact Analysis (k={results[0].k})', fontsize=14, fontweight='bold')
    
    ratios = [r.oversubscription_ratio for r in results]
    avg_throughputs = [r.average_throughput for r in results]
    p5_throughputs = [r.p5_throughput for r in results]
    p50_throughputs = [r.p50_throughput for r in results]
    p95_throughputs = [r.p95_throughput for r in results]
    
    # Plot 1: Throughput vs Oversubscription Ratio
    ax1 = axes[0, 0]
    ax1.plot(ratios, avg_throughputs, 'b-o', linewidth=2, markersize=8, label='Average')
    ax1.plot(ratios, p50_throughputs, 'g-s', linewidth=2, markersize=6, label='Median (P50)')
    ax1.plot(ratios, p5_throughputs, 'r-^', linewidth=2, markersize=6, label='5th Percentile')
    ax1.fill_between(ratios, p5_throughputs, p95_throughputs, alpha=0.2, color='blue', label='P5-P95 Range')
    ax1.set_xlabel('Oversubscription Ratio', fontsize=11)
    ax1.set_ylabel('Per-Flow Throughput (normalized)', fontsize=11)
    ax1.set_title('Throughput Degradation with Oversubscription', fontsize=12)
    ax1.set_xticks(ratios)
    ax1.set_xticklabels([f'{r}:1' for r in ratios])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentage reduction annotation
    if len(results) >= 2:
        baseline = results[0].average_throughput
        for i, (ratio, tp) in enumerate(zip(ratios[1:], avg_throughputs[1:]), 1):
            reduction = (baseline - tp) / baseline * 100
            ax1.annotate(f'-{reduction:.0f}%', (ratio, tp), 
                        textcoords="offset points", xytext=(5, 10), fontsize=9, color='red')
    
    # Plot 2: Throughput reduction percentage
    ax2 = axes[0, 1]
    baseline_throughput = results[0].average_throughput if results[0].average_throughput > 0 else 1
    reductions = [(baseline_throughput - r.average_throughput) / baseline_throughput * 100 for r in results]
    bars = ax2.bar(range(len(ratios)), reductions, color=['green' if r < 25 else 'orange' if r < 50 else 'red' for r in reductions], alpha=0.7)
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels([f'{r}:1' for r in ratios])
    ax2.set_xlabel('Oversubscription Ratio', fontsize=11)
    ax2.set_ylabel('Throughput Reduction (%)', fontsize=11)
    ax2.set_title('Throughput Loss vs Baseline (1:1)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, red in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{red:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Link Utilization CDF comparison
    ax3 = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, (r, color) in enumerate(zip(results, colors[:len(results)])):
        if r.link_utilizations:
            sorted_utils = sorted(r.link_utilizations)
            cdf = [(j + 1) / len(sorted_utils) for j in range(len(sorted_utils))]
            ax3.plot(sorted_utils, cdf, linewidth=2, color=color, 
                    label=f'{r.oversubscription_ratio}:1 ratio')
    ax3.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax3.set_xlabel('Link Utilization', fontsize=11)
    ax3.set_ylabel('CDF', fontsize=11)
    ax3.set_title('Link Utilization CDF', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.05)
    
    # Plot 4: Bottleneck links count
    ax4 = axes[1, 1]
    bottleneck_counts = [r.num_bottleneck_links for r in results]
    avg_utils = [r.avg_link_utilization * 100 for r in results]
    
    x = range(len(ratios))
    width = 0.35
    ax4.bar([i - width/2 for i in x], bottleneck_counts, width, label='Bottleneck Links (>80%)', color='red', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x, avg_utils, 'g-s', linewidth=2, markersize=8, label='Avg Utilization')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{r}:1' for r in ratios])
    ax4.set_xlabel('Oversubscription Ratio', fontsize=11)
    ax4.set_ylabel('Number of Bottleneck Links', fontsize=11, color='red')
    ax4_twin.set_ylabel('Average Link Utilization (%)', fontsize=11, color='green')
    ax4.set_title('Network Congestion Metrics', fontsize=12)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Oversubscription analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_oversubscription_throughput_drop(
    results: List[OversubscriptionResult],
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Generate a dedicated plot for Section 4.3 showing throughput degradation
    as oversubscription ratio increases from 1:1 to 8:1.
    
    This visualization is designed for presentation/documentation purposes,
    supporting quantitative impact findings in Section 5.2.
    
    Args:
        results: List of OversubscriptionResult from simulation
        save_path: Path to save the figurae (e.g., "section_4_3_oversubscription.png")
        show_plot: Whether to display interactively
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    if not results:
        print("No results to plot.")
        return
    
    # Extract data
    ratios = [r.oversubscription_ratio for r in results]
    avg_throughputs = [r.average_throughput for r in results]
    p5_throughputs = [r.p5_throughput for r in results]
    p50_throughputs = [r.p50_throughput for r in results]
    p95_throughputs = [r.p95_throughput for r in results]
    
    # Calculate percent of baseline
    baseline = results[0].average_throughput if results[0].average_throughput > 0 else 1.0
    throughput_pct = [(t / baseline) * 100 for t in avg_throughputs]
    p5_pct = [(t / baseline) * 100 for t in p5_throughputs]
    p50_pct = [(t / baseline) * 100 for t in p50_throughputs]
    p95_pct = [(t / baseline) * 100 for t in p95_throughputs]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 4.3: Oversubscription Impact on Network Throughput', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # =========================================================================
    # LEFT PLOT: Throughput Drop (Absolute Values)
    # =========================================================================
    
    # Main throughput line with confidence band
    ax1.fill_between(ratios, p5_throughputs, p95_throughputs, 
                     alpha=0.25, color='steelblue', label='P5-P95 Range')
    ax1.plot(ratios, avg_throughputs, 'o-', color='navy', linewidth=2.5, 
             markersize=10, label='Average Throughput', zorder=5)
    ax1.plot(ratios, p50_throughputs, 's--', color='forestgreen', linewidth=2, 
             markersize=7, label='Median (P50)', zorder=4)
    ax1.plot(ratios, p5_throughputs, '^:', color='crimson', linewidth=2, 
             markersize=7, label='5th Percentile (Tail)', zorder=4)
    
    # Annotations showing percentage reduction from baseline
    for i, (ratio, tp) in enumerate(zip(ratios, avg_throughputs)):
        if i > 0:  # Skip baseline
            reduction = ((baseline - tp) / baseline) * 100
            ax1.annotate(f'-{reduction:.0f}%', 
                        xy=(ratio, tp), 
                        xytext=(8, 12),
                        textcoords='offset points',
                        fontsize=11, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                 edgecolor='orange', alpha=0.9))
    
    ax1.set_xlabel('Oversubscription Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Per-Flow Throughput (Normalized)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput Degradation Curve', fontsize=12)
    ax1.set_xticks(ratios)
    ax1.set_xticklabels([f'{int(r)}:1' for r in ratios], fontsize=11)
    ax1.set_ylim(0, max(avg_throughputs) * 1.15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference line at baseline
    ax1.axhline(y=baseline, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(ratios[-1], baseline * 1.02, 'Baseline (1:1)', 
             fontsize=9, color='gray', ha='right')
    
    # =========================================================================
    # RIGHT PLOT: Bar Chart - Percentage of Baseline Throughput
    # =========================================================================
    
    x_pos = range(len(ratios))
    bar_width = 0.6
    
    # Color gradient from green to red based on throughput retention
    colors = []
    for pct in throughput_pct:
        if pct >= 80:
            colors.append('#2ecc71')  # Green
        elif pct >= 60:
            colors.append('#f39c12')  # Orange
        elif pct >= 40:
            colors.append('#e67e22')  # Dark Orange
        else:
            colors.append('#e74c3c')  # Red
    
    bars = ax2.bar(x_pos, throughput_pct, bar_width, color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, throughput_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{pct:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Add tail latency markers (P5)
    for i, (pos, p5) in enumerate(zip(x_pos, p5_pct)):
        ax2.plot(pos, p5, 'v', color='darkred', markersize=10, zorder=5)
        if i > 0:  # Show P5 value for non-baseline
            ax2.annotate(f'P5: {p5:.1f}%', 
                        xy=(pos, p5),
                        xytext=(0, -18),
                        textcoords='offset points',
                        fontsize=9, color='darkred', ha='center')
    
    ax2.set_xlabel('Oversubscription Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (% of 1:1 Baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Retention by Ratio', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{int(r)}:1' for r in ratios], fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.axhline(y=100, color='navy', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (100%)')
    ax2.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5, label='50% Threshold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add legend for P5 marker
    ax2.plot([], [], 'v', color='darkred', markersize=8, label='5th Percentile')
    ax2.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Summary Text Box
    # =========================================================================
    
    # Calculate key metrics for summary
    if len(results) >= 2:
        ratio_8_result = next((r for r in results if r.oversubscription_ratio >= 8.0), results[-1])
        max_reduction = ((baseline - ratio_8_result.average_throughput) / baseline) * 100
        worst_p5 = (ratio_8_result.p5_throughput / baseline) * 100
        
        summary_text = (
            f"Key Findings:\n"
            f"• 1:1 → 8:1: {max_reduction:.0f}% throughput loss\n"
            f"• Worst-case (P5) at 8:1: {worst_p5:.0f}% of baseline\n"
            f"• Flows: {results[0].num_flows} | k={results[0].k}"
        )
        
        fig.text(0.5, -0.08, summary_text, ha='center', va='top',
                fontsize=11, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Section 4.3 oversubscription throughput plot saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_throughput_histogram(
    results: List[OversubscriptionResult],
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    Plot histogram of per-flow throughput for different oversubscription ratios.
    
    Shows the distribution shift and long-tail behavior as oversubscription increases.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    # Filter to show 1:1 vs 4:1 comparison primarily
    ratios_to_show = [1.0, 4.0]
    results_to_show = [r for r in results if r.oversubscription_ratio in ratios_to_show]
    
    if len(results_to_show) < 2:
        results_to_show = results[:2] if len(results) >= 2 else results
    
    fig, axes = plt.subplots(1, len(results_to_show), figsize=(6*len(results_to_show), 5))
    if len(results_to_show) == 1:
        axes = [axes]
    
    fig.suptitle('Per-Flow Throughput Distribution', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for ax, r, color in zip(axes, results_to_show, colors):
        throughputs = r.throughput_per_flow
        if throughputs:
            ax.hist(throughputs, bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(r.average_throughput, color='black', linestyle='--', linewidth=2, label=f'Mean: {r.average_throughput:.4f}')
            ax.axvline(r.p5_throughput, color='red', linestyle=':', linewidth=2, label=f'P5: {r.p5_throughput:.4f}')
            ax.set_xlabel('Per-Flow Throughput', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{r.oversubscription_ratio}:1 Oversubscription', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Throughput histogram saved to: {save_path}")
    
    if show_plot:
        plt.show()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class ScalingExperiment:
    """
    Container for running and storing scaling experiments.
    
    Supports multiple k values and export to various formats.
    """
    k_values: List[int]
    link_capacity_gbps: float = 10.0
    results: List[FatTreeMetrics] = field(default_factory=list)
    
    def run(self) -> "ScalingExperiment":
        """Execute the experiment for all k values."""
        self.results = [
            compute_fat_tree_metrics(k, self.link_capacity_gbps)
            for k in self.k_values
        ]
        return self
    
    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export results to JSON."""
        data = [r.to_dict() for r in self.results]
        json_str = json.dumps(data, indent=2)
        if filepath:
            filepath.write_text(json_str)
        return json_str
    
    def to_csv(self, filepath: Path) -> None:
        """Export results to CSV file."""
        if not self.results:
            raise ValueError("No results to export. Run experiment first.")
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
    
    def summary_table(self) -> str:
        """Generate ASCII table summary."""
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


def compare_topologies(
    metrics_a: FatTreeMetrics,
    metrics_b: FatTreeMetrics
) -> dict:
    """
    Compare two fat-tree configurations.
    
    Args:
        metrics_a: First topology metrics
        metrics_b: Second topology metrics
        
    Returns:
        Dictionary with comparison ratios and differences
    """
    return {
        "k_ratio": metrics_b.k / metrics_a.k,
        "host_ratio": metrics_b.hosts / metrics_a.hosts,
        "switch_ratio": metrics_b.total_switches / metrics_a.total_switches,
        "bandwidth_ratio": metrics_b.bisection_bandwidth_gbps / metrics_a.bisection_bandwidth_gbps,
        "host_increase": metrics_b.hosts - metrics_a.hosts,
        "switch_increase": metrics_b.total_switches - metrics_a.total_switches,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point demonstrating all module capabilities."""
    
    print("=" * 70)
    print("  FAT-TREE TOPOLOGY COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. SCALING EXPERIMENT
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 1: SCALING ANALYSIS")
    print("─" * 70)
    
    experiment = ScalingExperiment(
        k_values=[4, 8, 24, 48],
        link_capacity_gbps=10.0
    ).run()
    
    print("\n" + experiment.summary_table())
    
    # -------------------------------------------------------------------------
    # 2. DETAILED METRICS
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 2: DETAILED METRICS")
    print("─" * 70)
    
    for metrics in experiment.results:
        print(f"\n{metrics}")
    
    # -------------------------------------------------------------------------
    # 3. COST ESTIMATION
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 3: COST ESTIMATION")
    print("─" * 70)
    
    pricing = SwitchPricing()
    print(f"\n{pricing}\n")
    
    for metrics in experiment.results:
        cost = estimate_network_cost(metrics, pricing)
        print(f"\n--- k={metrics.k} ---")
        print(cost)
    
    # -------------------------------------------------------------------------
    # 4. FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 4: FAILURE ANALYSIS")
    print("─" * 70)
    
    for k in [4, 24]:
        analysis = analyze_failure_impact(k)
        print(f"\n{analysis}")
    
    # Cascading failure simulation
    print("\n--- Cascading Failure Simulation (k=24) ---")
    scenarios = [
        {"failed_core_switches": 5},
        {"failed_edge_switches": 10},
        {"failed_core_switches": 10, "failed_agg_switches": 20},
    ]
    
    for scenario in scenarios:
        result = simulate_cascading_failures(24, **scenario)
        print(f"\nScenario: {scenario}")
        print(f"  Status: {result['status']}")
        print(f"  Hosts connected: {result['connectivity_pct']:.1f}%")
        print(f"  Bisection BW remaining: {result['bisection_bandwidth_remaining_pct']:.1f}%")
    
    # -------------------------------------------------------------------------
    # 5. OVERSUBSCRIPTION SIMULATION (NEW!)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 5: OVERSUBSCRIPTION SIMULATION")
    print("─" * 70)
    
    print("\n5.1 Running oversubscription experiments...")
    
    # Run simulation for k=8 (128 hosts) with different ratios
    oversub_results = run_oversubscription_experiment(
        k=8,
        ratios=[1.0, 2.0, 4.0, 8.0],
        num_flows=200,
        traffic_pattern="random",
        seed=42
    )
    
    # Print detailed results for each ratio
    for result in oversub_results:
        print(f"\n{result}")
    
    # Print comparison table
    print("\n5.2 Comparison across oversubscription ratios:")
    print(print_oversubscription_comparison(oversub_results))
    
    # Test with inter-pod traffic pattern (more demanding)
    print("\n5.3 Inter-pod traffic pattern simulation:")
    interpod_results = run_oversubscription_experiment(
        k=8,
        ratios=[1.0, 4.0],
        num_flows=150,
        traffic_pattern="inter_pod",
        seed=42
    )
    print(print_oversubscription_comparison(interpod_results))
    
    # -------------------------------------------------------------------------
    # 6. VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 6: VISUALIZATION")
    print("─" * 70)
    
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        try:
            # Uncomment to show plots interactively:
            plot_cubic_scaling(show_plot=True, save_path=Path("fat_tree_scaling.png"))
            plot_cost_analysis(show_plot=True, save_path=Path("fat_tree_costs.png"))
            plot_failure_resilience(k=24, show_plot=True, save_path=Path("fat_tree_resilience.png"))
            
            # NEW: Oversubscription visualizations
            plot_oversubscription_analysis(oversub_results, show_plot=True, save_path=Path("fat_tree_oversubscription.png"))
            plot_throughput_histogram(oversub_results, show_plot=True, save_path=Path("fat_tree_throughput_dist.png"))
            
        except Exception as e:
            print(f"  Plotting error: {e}")
            print("  (Plots may require a display environment)")
    else:
        print("\n  matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")
    
    # -------------------------------------------------------------------------
    # TOPOLOGY COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  TOPOLOGY COMPARISON (k=4 vs k=48)")
    print("─" * 70)
    
    small = experiment.results[0]
    large = experiment.results[-1]
    comparison = compare_topologies(small, large)
    
    print(f"\n  Host scaling:      {comparison['host_ratio']:,.0f}x")
    print(f"  Switch scaling:    {comparison['switch_ratio']:.1f}x")
    print(f"  Bandwidth scaling: {comparison['bandwidth_ratio']:,.0f}x")
    
    # -------------------------------------------------------------------------
    # QUANTITATIVE IMPACT SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  SECTION 5.3: QUANTITATIVE IMPACT OF OVERSUBSCRIPTION")
    print("─" * 70)
    
    if len(oversub_results) >= 2:
        ratio_1 = oversub_results[0]
        ratio_4 = next((r for r in oversub_results if r.oversubscription_ratio >= 4.0), None)
        
        if ratio_4:
            throughput_factor = ratio_4.average_throughput / ratio_1.average_throughput if ratio_1.average_throughput > 0 else 0
            throughput_reduction_pct = (1 - throughput_factor) * 100
            p5_pct_of_line_rate = ratio_4.p5_throughput * 100
            
            print(f"""
    QUANTITATIVE FINDINGS:
    
    In our simulation, increasing oversubscription from 1:1 to 4:1 in a k={ratio_1.k} 
    fat-tree reduced average per-flow throughput by a factor of approximately 
    {throughput_factor:.2f}x ({throughput_reduction_pct:.1f}% reduction).
    
    The simulation introduced a long tail of flows experiencing less than 
    {p5_pct_of_line_rate:.1f}% of line rate (5th percentile throughput).
    
    This behavior mirrors reports from multi-tenant training environments where 
    oversubscription and background traffic substantially degrade training throughput.
    
    Key Statistics:
      - Baseline (1:1) avg throughput: {ratio_1.average_throughput:.4f}
      - 4:1 oversubscription avg:      {ratio_4.average_throughput:.4f}
      - 4:1 P5 (worst 5%):             {ratio_4.p5_throughput:.4f}
      - 4:1 bottleneck links:          {ratio_4.num_bottleneck_links}
            """)
    
    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
