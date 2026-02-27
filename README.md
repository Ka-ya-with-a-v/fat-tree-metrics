# fat-tree-metrics Project Overview



This project investigates the structural scalability and performance characteristics of k-ary Fat-Tree (Clos) network topologies, which form the backbone of modern hyperscale data centers. As AI workloads increasingly dominate cloud infrastructure, network design has shifted toward architectures that provide high path diversity and near full bisection bandwidth to support large-scale distributed training.

The study begins with analytical derivations of key combinatorial properties of a k-ary Fat-Tree, including total host count, number of core switches, inter-pod equal-cost paths, and bisection bandwidth. Closed-form expressions are derived and validated for multiple values of 
𝑘
k, demonstrating cubic scaling of both host capacity and aggregate bandwidth. These results highlight why Fat-Tree topologies remain attractive for GPU-intensive AI clusters.

To complement theoretical analysis, a parameterized Python implementation was developed to compute topology metrics and validate analytical formulas. Scaling experiments were conducted for increasing values of 
𝑘
k, and visualization scripts were used to illustrate structural layout and performance trends. An additional focused experiment explores the impact of oversubscription ratios on effective throughput, providing quantitative insight into the trade-off between cost and performance in real-world deployments.

The project connects classical network topology theory with contemporary AI-era system demands. While the combinatorial properties of Fat-Trees are well established, this work frames them within the context of distributed machine learning workloads, emphasizing the practical implications of path redundancy and bisection bandwidth on training efficiency.
