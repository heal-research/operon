Features
========

.. _implementation_details:

Implementation details
----------------------

* Compact, efficient linear encoding for trees. 
* Direct correspondence to GP tree concept via linear (postfix) indexing scheme.
* Trees represented as contiguous node arrays with 40 bytes per tree node, promoting memory locality.
* Low memory footprint: 10k trees of length 50 (20k internally for parent and offspring populations) in under 20MiB of memory.
* Logical parallelism: *recombinants* (new offspring) are generated concurrently.
* The framework handles threads and scheduling.
* Low-overhead synchronization via atomic primitives.
* Ability to evolve very large populations (eg., 1M individuals on standard hardware)
* Designed as a core library (``liboperon``) and a CLI client (``operon-gp``) easily integrated into eg. Python or shell scripts

.. _gp_design:

Genetic programming design
--------------------------

* Multiple GP flavors sharing the same evolutionary main loop, differing in how they define an offspring generator concept
* An offspring generator may fail (returns an option type) for configurable reasons such as offspring acceptance criteria.
* Streaming genetic operators designed to integrate efficiently with the concurrency model.
* Operators spend from the global evaluation budget,,making it easy to implement fair comparisons between algorithms.
* Operators encapsulate termination criteria (eg., budget based, selection pressure based, etc.).
* State-of-the-art, numerically-stable statistical methods (mean, variance, correlation).
* Seamless integration with automatic differentiation (same evaluation can work on both scalar and dual number types).
* Local search support (trust region-based fitting of tree numerical coefficients, also counted out of the global evaluation budget)
* Detailed local search statistics (number of Jacobian evaluations, failed/successful gradient descent steps, convergence status).
* Ability to perform model evaluation in single- or double precision mode (eg., using float (32-bit) or double (64-bit) data types).
* Novel tree initialization (balanced tree creator) able to produce arbitrary distributions of tree sizes and symbol frequencies.
