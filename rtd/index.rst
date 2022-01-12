.. operon documentation master file, created by
   sphinx-quickstart on Sun Apr 19 16:39:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: _static/logo_mini.png
    :height: 100px
    :align: center


.. note::
    **The documentation is under construction.**

Introduction
============

*Operon* is a modern C++ framework for `symbolic regression <https://en.wikipedia.org/wiki/Symbolic_regression>`_ that uses `genetic programming <https://en.wikipedia.org/wiki/Genetic_programming>`_ to explore a hypothesis space of possible mathematical expressions in order to find the best-fitting model for a given `regression target <https://en.wikipedia.org/wiki/Regression_analysis>`_.
Its main purpose is to help develop accurate and interpretable white-box models in areas such as `system identification <https://en.wikipedia.org/wiki/System_identification>`_.

.. image:: _static/evo_rtd.gif
    :align: center

Motivation
----------

*Operon* was motivated by the need to have a *flexible* and *performant* system that *works out of the box*. Thus, it was developed with the following goals in mind:

Modern concurrency model
    Traditional threading approaches are not optimal for today's many-core systems. This means designing the evolutionary main loop in such a way as to avoid synchronisation overhead and take advantage of C++17's `execution policies`_.  

Performance
    By using an efficient linear tree representation where each ``Node`` is `trivial`_ and vectorized evaluation with the help of the `Eigen <https://eigen.tuxfamily.org/>`_ library. The encoding consumes 40 bytes per tree node, allowing practitioners to work with very large populations.

Ease-of-use
    *Operon* (the core library) comes with a command-line client that just works: you pass it a dataset and it will start optimizing. Its behavior can be configured by command line options, making it easy to integrate with any scripting environment or high-level language such as Python. A Python script is provided for performing experiments automatically aggregating the results. 

    For more advanced use cases, we provide a C++ and a Python API, briefly illustrated with some `examples <https://github.com/foolnotion/operon/tree/master/examples>`_.

For an overview of *Operon* please have a look at the :doc:`features` page. 

The software was also presented at GECCO'2020 *EvoSoft* workshop: https://dl.acm.org/doi/10.1145/3377929.3398099. If you want to reference it in your publication, please use:

Reference
^^^^^^^^^

.. code-block:: tex

    @inproceedings{Burlacu:2020:GECCOcomp,
    author = {Bogdan Burlacu and Gabriel Kronberger and Michael Kommenda},
    title = {Operon C++: An Efficient Genetic Programming Framework for Symbolic Regression},
    year = {2020},
      editor = {Richard Allmendinger and Hugo Terashima Marin and Efren Mezura Montes and Thomas Bartz-Beielstein and Bogdan Filipic and Ke Tang and David Howard and Emma Hart and Gusz Eiben and Tome Eftimov and William {La Cava} and Boris Naujoks and Pietro Oliveto and Vanessa Volz and Thomas Weise and Bilel Derbel and Ke Li and Xiaodong Li and Saul Zapotecas and Qingfu Zhang and Rui Wang and Ran Cheng and Guohua Wu and Miqing Li and Hisao Ishibuchi and Jonathan Fieldsend and Ozgur Akman and Khulood Alyahya and Juergen Branke and John R. Woodward and Daniel R. Tauritz and Marco Baioletti and Josu Ceberio Uribe and John McCall and Alfredo Milani and Stefan Wagner and Michael Affenzeller and Bradley Alexander and Alexander (Sandy) Brownlee and Saemundur O. Haraldsson and Markus Wagner and Nayat Sanchez-Pi and Luis Marti and Silvino {Fernandez Alzueta} and Pablo {Valledor Pellicer} and Thomas Stuetzle and Matthew Johns and Nick Ross and Ed Keedwell and Herman Mahmoud and David Walker and Anthony Stein and Masaya Nakata and David Paetzel and Neil Vaughan and Stephen Smith and Stefano Cagnoni and Robert M. Patton and Ivanoe {De Falco} and Antonio {Della Cioppa} and Umberto Scafuri and Ernesto Tarantino and Akira Oyama and Koji Shimoyama and Hemant Kumar Singh and Kazuhisa Chiba and Pramudita Satria Palar and Alma Rahat and Richard Everson and Handing Wang and Yaochu Jin and Erik Hemberg and Riyad Alshammari and Tokunbo Makanju and Fuijimino-shi and Ivan Zelinka and Swagatam Das and Ponnuthurai Nagaratnam and Roman Senkerik},
      isbn13 = {9781450371278},
    publisher = {Association for Computing Machinery},
      publisher_address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3377929.3398099},
    doi = {doi:10.1145/3377929.3398099},
    booktitle = {Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
    pages = {1562–1570},
      size = {9 pages},
      keywords = {genetic algorithms, genetic programming, C++, symbolic regression},
      address = {internet},
    series = {GECCO '20},
      month = {July 8-12},
      organisation = {SIGEVO},
      abstract = {},
      notes = {Also known as \cite{10.1145/3377929.3398099}
               GECCO-2020
               A Recombination of the 29th International Conference on Genetic Algorithms (ICGA) and the 25th Annual Genetic Programming Conference (GP)},
    }

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
    :maxdepth: 0

    self

    features
    build
    example

.. _`symbolic regression`: https://en.wikipedia.org/wiki/Symbolic_regression
.. _`genetic programming`: https://en.wikipedia.org/wiki/Genetic_programming
.. _`execution policies`: https://www.bfilipek.com/2018/11/parallel-alg-perf.html
.. _`trivial`: https://en.cppreference.com/w/cpp/named_req/TrivialType
.. _'Eigen`: http://eigen.tuxfamily.org/index.php?title=Main_Page
