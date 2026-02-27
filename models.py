from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum


class FatTreeSize(Enum):
    SMALL = 4
    MEDIUM = 8
    LARGE = 24
    XLARGE = 48


@dataclass
class SwitchPricing:
    core_switch_price: float = 50_000.0
    aggregation_switch_price: float = 25_000.0
    edge_switch_price: float = 5_000.0
    link_cost_per_gbps: float = 100.0

    def __str__(self) -> str:
        return (
            f"Switch Pricing:\n"
            f"  Core: ${self.core_switch_price:,.0f}\n"
            f"  Aggregation: ${self.aggregation_switch_price:,.0f}\n"
            f"  Edge: ${self.edge_switch_price:,.0f}\n"
            f"  Link cost/Gbps: ${self.link_cost_per_gbps:,.0f}"
        )


@dataclass(frozen=True, slots=True)
class FatTreeMetrics:
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
        return asdict(self)


@dataclass
class CostEstimate:
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
    k: int
    single_core_failure_host_pairs_affected_pct: float
    single_agg_failure_host_pairs_affected_pct: float
    single_edge_failure_hosts_disconnected: int
    single_link_redundancy: str
    intrapod_paths: int
    interpod_paths: int
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


class FatTreeValidationError(Exception):
    pass
