.. operon documentation master file, created by
   sphinx-quickstart on Sun Apr 19 16:39:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

*Operon* is a `genetic programming`_ (GP) framework for `symbolic regression`_ (SR) implemented in modern C++ with a focus on efficiency and performance. 

Motivation
----------

*Operon* was motivated by the need to have a *flexible* and *performant* system that *works out of the box*. Thus, it was developed with the following goals in mind:

Modern concurrency model
    Traditional threading approaches are not optimal for today's many-core systems. This means designing the evolutionary main loop in such a way as to avoid synchronisation overhead and take advantage of C++17's `execution policies`_.  

Performance
    By using an efficient linear tree representation where each ``Node`` is `trivial`_ and vectorized evaluation with the help of the `Eigen_` library. The encoding consumes 40 bytes per tree node, allowing practitioners to work with very large populations.

Ease-of-use
    We assume the user wants to put the framework to work with some real data and does not wish to waste time with useless tutorials and toy examples. 

    *Operon* (the core library) is complemented by a command-line client that just works: you pass it a dataset and it will start optimizing. Its behavior can be configured by command line options, making it easy to integrate with any scripting environment or high-level language such as Python. A Python script is provided for performing experiments automatically aggregating the results. 

For an overview of *Operon* please have a look at the :doc:`features` page. The API is illustrated by an :doc:`example`.
The main concepts are described in the :doc:`operators` section. 

.. .. toctree::
..     :maxdepth: 0
..     :caption: Contents:
.. 
..     features 
..     ...
.. 
.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. toctree::
    :maxdepth: 1
    :hidden:

    features
    example

.. _`symbolic regression`: https://en.wikipedia.org/wiki/Symbolic_regression
.. _`genetic programming`: https://en.wikipedia.org/wiki/Genetic_programming
.. _`execution policies`: https://www.bfilipek.com/2018/11/parallel-alg-perf.html
.. _`trivial`: https://en.cppreference.com/w/cpp/named_req/TrivialType
.. _'Eigen`: http://eigen.tuxfamily.org/index.php?title=Main_Page
