# PHYS170X Project: CGnets (Sp2025).
Codebase for CGnet implementation in Julia for HMC PHYS170X created by Ananya and Paco. 

We have also included previous versions of CGnets, as well as write-ups and supporting files, in the repository. We aimed to reproduce force-matching and PMF figures from the paper "Machine Learning of Coarse-Grained Molecular Dynamics Force Fields," which may be found at https://pubs.acs.org/doi/10.1021/acscentsci.8b00913.

* **`CGnets_full.jl`**: Implements the full CGnet architecture following the approach described in the paper, including energy-based training and force computation via automatic differentiation. This version adheres closely to the original formulation but can be challenging to optimize.

* **`CGnets_simplified.jl`**: A streamlined version that bypasses the energy formulation by directly training the network to predict forces. While conceptually simpler and less physically grounded, this model achieves significantly better empirical performance.
