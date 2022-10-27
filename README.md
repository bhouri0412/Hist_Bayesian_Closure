## Hist_Bayesian_Closure

Code and data accompanying the manuscript titled "History-Based, Bayesian, Closure for Stochastic Parameterization: Application to Lorenz '96", authored by Mohamed Aziz Bhouri and Pierre Gentine.

## Abstract

Physical parameterizations are used as representations of unresolved subgrid processes within weather and global climate models or coarse-scale turbulent models, whose resolutions are too coarse to resolve small-scale processes. These parameterizations are typically grounded on physically-based, yet empirical, representations of the underlying small-scale processes. Machine learning-based parameterizations have recently been proposed as an alternative and have shown great promises to reduce uncertainties associated with small-scale processes. Yet, those approaches still show some important mismatches that are often attributed to stochasticity in the considered process. This stochasticity can be due to noisy data, unresolved variables or simply to the inherent chaotic nature of the process. To address these issues, we develop a new type of parameterization (closure) which is based on a Bayesian formalism for neural networks, to account for uncertainty quantification, and includes memory, to account for the non-instantaneous response of the closure. To overcome the curse of dimensionality of Bayesian techniques in high-dimensional spaces, the Bayesian strategy is based on a Hamiltonian Monte Carlo Markov Chain sampling strategy that takes advantage of the likelihood function and kinetic energy's gradients with respect to the parameters to accelerate the sampling process. We apply the proposed Bayesian history-based parameterization to the Lorenz '96 model in the presence of noisy and sparse data, similar to satellite observations, and show its capacity to predict skillful forecasts of the resolved variables while returning trustworthy uncertainty quantifications for different sources of error. This approach paves the way for the use of Bayesian approaches for closure problems.

## Citation

    @article{Bhouri2022HistParam,
    title={History-Based, Bayesian, Closure for Stochastic Parameterization: Application to Lorenz '96},
    author={Bhouri, Mohamed Aziz  and Gentine, Pierre},
    journal={arXiv preprint arXiv:2210.14488},
    doi = {https://doi.org/10.48550/arXiv.2210.14488},
    year={2022}
  }
