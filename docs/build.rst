
Build instructions 
==================

The project requires ``CMake`` and a ``C++17`` compliant compiler that supports execution policies from ``std::execution``. 
We rely on a number of libraries, the majority of which should be accessible via your distribution's package manager.  On Windows they can be easily managed using `vcpkg <https://github.com/Microsoft/vcpkg>`_. A few small header-only libraries are downloaded by ``CMake`` during the build generation phase.

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- `Threading Building Blocks <https://github.com/oneapi-src/oneTBB>`_ ― the backend for Operon's concurrency model.
- `Eigen <http://eigen.tuxfamily.org>`_ ― used internally for model evaluation.  
- `Ceres <http://ceres-solver.org>`_ ― for numerical fitting of model coefficients. 
- `{fmt} <https://fmt.dev/latest/index.html>`_ ― used internally instead of iostreams. 

.. note::
    Using the git versions of ``Eigen`` and ``Ceres`` is **strongly** recommended. On Windows we recommend building with ``MinGW`` or using the Windows Subsystem for Linux (``WSL``).

Automatically downloaded by CMake
"""""""""""""""""""""""""""""""""
- `microsoft-gsl <https://github.com/microsoft/GSL>`_ ― Microsoft's implementation of the `C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_
- `rapidcsv <https://github.com/d99kris/rapidcsv>`_ ― header-only library for CSV parsing 
- `nanobench <https://github.com/martinus/nanobench>`_ ― microbenchmarking library used in the unit tests
- `xxhash <https://github.com/Cyan4973/xxHash>`_ ― we use the header-only version for tree hashing 

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

- `cxxopts <https://github.com/jarro2783/cxxopts>`_ ― required for the cli app.
- `doctest <https://github.com/onqtam/doctest>`_ ― required for unit tests.
- `python <https://www.python.org/>`_ and `pybind11 <https://github.com/pybind/pybind11>`_ ― required to build the python bindings.

Configuration options
^^^^^^^^^^^^^^^^^^^^^

.. note::
    These options are specified to ``CMake`` in the form ``-D<OPTION>=<ON|OFF>``. All options are ``OFF`` by default. Options that depend on additional libraries require those libraries to be present and detectable ``CMake``. 

#. ``USE_SINGLE_PRECISION``: Perform model evaluation using floats (single precision) instead of doubles. Great for reducing runtime, might not be appropriate for all purposes. 
#. ``USE_OPENLIBM``: Link against Julia's openlibm, a high performance mathematical library (recommended to improve consistency across compilers and operating systems).
#. ``BUILD_TESTS``: Build the unit tests.
#. ``BUILD_PYBIND``: Build the Python bindings.
#. ``USE_JEMALLOC``: Link against `jemalloc <http://jemalloc.net/>`_, a general purpose ``malloc(3)`` implementation that emphasizes fragmentation avoidance and scalable concurrency support (mutually exclusive with ``tcmalloc``).
#. ``USE_TCMALLOC``: Link against `tcmalloc <https://google.github.io/tcmalloc/>`_ (thread-caching malloc), a ``malloc(3)`` implementation that reduces lock contention for multi-threaded programs (mutually exclusive with `jemalloc`).

Install Examples
^^^^^^^^^^^^^^^^

Here is an example install of operon in linux using a `conda <https://anaconda.org/anaconda/conda>` environment and bash commands.

1. Specify an `environment.yml` file with the dependencies:

   .. code-block:: yaml

      name: operon-env
      channels:
        - conda-forge
      dependencies:
        - python=3.9.1
        - cmake=3.19.1  
        - pybind11=2.6.1 
        - eigen=3.3.9 
        - fmt=7.1.3 
        - ceres-solver=2.0.0 
        - tbb-devel=2020.2 
        - openlibm 
        - cxxopts 

2. Run the following commands:

   .. code-block:: Python
        
      # create and activate conda environment
      conda env create -f environment.yml
      conda activate operon-env

      # Use gcc-9
      export CC=gcc-9
      export CXX=gcc-9

      # clone operon
      git clone https://github.com/heal-research/operon
      cd operon

      # run cmake with options
      mkdir build; cd build; 
      cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON 

      # build
      make VERBOSE=1 -j pyoperon

      # install python package
      make install

3. To test that the python package installed correctly, try `python -c "from operon.sklearn import SymbolicRegressor`. 
