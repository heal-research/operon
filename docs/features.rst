Features
========

Genetic programming is quite computationally intensive, but that doesn't mean it has to be slow. 
*Operon*'s main goals are to provide excellent performance and to offer a user-friendly experience.

Implementation
--------------

* Cross-platform
* Highly-scalable concurrency model using `Threading Building Blocks <https://github.com/oneapi-src/oneTBB>`_
* Low synchronization overhead via atomic primitives
* Very fast, vectorized model evaluation using `Eigen <https://eigen.tuxfamily.org/>`_
* Low memory footprint (linear tree encoding as an array of ``Node`` (40 byte each)
* Support for numerical and automatic differentiation using `Ceres <http://ceres-solver.org/>`_
* Nonlinear least squares optimization of model parameters using `Ceres <http://ceres-solver.org/>`_
* State of the art numerically stable error metrics (R-Squared, MSE, RMSE, NMSE)
* Python bindings using `pybind11 <https://github.com/pybind/pybind11>`_
* Modern genetic programming design (see below)

Encoding
--------

* *Operon* uses a linear postfix representation for expression trees
* A ``Tree`` encapsulates a contiguous array of ``Nodes``. 
* The ``Node`` is ``trivial`` and has ``standard_layout``.
* All operators manipulate the representation directly, with no intermediate step.

Logical parallelism
-------------------

* *Streaming* genetic operators designed to integrate efficiently with the concurrency model
* New individuals are generated concurrently in separate logical threads
* Optimal distribution of logical tasks to physical threads is left to the underlying scheduler
* Consistency and predictability across different machines (to the extent possible)


Evolutionary model
------------------

*Operon* implements an efficient evolutionary model centered around the concept of an *offspring generator* ― an operator that encapsulates a preconfigured recipe for producing a single child individidual from the parent population. A general recipe would look like below: 

selection ⟼ crossover ⟼ mutation ⟼ evaluation ⟼ acceptance

Because these operators don't share mutable state, the execution of such pipelines can be easily parallelized, such that each offspring generation event takes place independently in its own *logical thread*.

The *offspring generator* concept allows deploying different evolutionary models (standard, offspring selection, soft brood selection) within the same algorithmic main loop.

Hybridization with local search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Conceptually, this kind of hybridization is useful to shift the effort of finding appropriate numerical coefficients for the model from the evolutionary algorithm itself to a specialized optimization procedure. Algorithms that incorporate some kind of local search optimization step are also called *memetic algorithms*. 

Under the hood, we take advantage of `Ceres <http://ceres-solver.org/>`_' integration with `Eigen <https://eigen.tuxfamily.org/>`_ in order to optionally perform non-linear least squares fitting of model coefficients during the model evaluation step. The local optimization step is typically quite computationally intensive since model derivatives have to be computed (Ceres supports numerical or automatic differentiation).

In order to promote fair comparison between different algorithmic variants (with and without local search), we allow constraints on an algorithm's evaluation budget -- including local search. 

Genetic operators
-----------------
Operators spend from a global evaluation budget, facilitating fair comparisons between algorithmic flavors. All the typical selection and recombination opperations are supported:

Selection
^^^^^^^^^
**Random**, **proportional** and **tournament** selection are suported. In the case of crossover, a separate selection mechanism can be configured for each parent.

Mutation
^^^^^^^^

Mutation can act on a single node or on an entire subtree. Depending on the node type, single-node mutation can change the node value, the data label or the type of mathematical operation performed by the node. Subtree mutation can insert, replace or delete a subtree.

Crossover
^^^^^^^^^

The crossover operator generates a single child individual from two parents, by swapping a subtree from the first parent (algo called the *root parent*) with a subtree from the second parent (also called the *non-root parent*). The operator supports a configurable bias towards internal nodes.  

Both **crossover** and **mutation** operators are designed to ensure that depth and length restrictions on the tree individuals are always respected.


Tree initialization
^^^^^^^^^^^^^^^^^^^

Supported algorithms: 

- Grow tree creator 
- Balanced tree creator (BTC)
- Probabilistic tree creator (PTC)
 
* Configurable primitive set

* Hybridization with local search
* Novel tree hashing algorithm (`paper <https://dblp.org/rec/journals/corr/abs-1902-00882>`_)
* Fast calculation of population diversity (using tree hashes)

.. rubric:: Footnotes
.. [#] https://docs.microsoft.com/en-us/cpp/cpp/trivial-standard-layout-and-pod-types
.. [#] `Jets/Dual numbers <http://ceres-solver.org/automatic_derivatives.html#dual-numbers-jets>`_ provided by `Ceres Solver <http://ceres-solver.org>`_
.. [#] Arbitrary precision support using `MPFR <https://www.mpfr.org>`_ via its C++ interface `MPFR C++ <http://www.holoborodko.com/pavel/mpfr/>`_
