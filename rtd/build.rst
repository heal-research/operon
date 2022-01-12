
Build instructions 
==================

The project requires ``CMake`` and a ``C++17`` compliant compiler.

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

Linux/Conda
"""""""""""

Here is an example install of operon in linux using a `conda <https://anaconda.org/anaconda/conda>`_ environment and bash commands.

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
        - taskflow=3.1.0
        - openlibm 
        - cxxopts 

2. Run the following commands:

   .. code-block:: Python
        
      # create and activate conda environment
      conda env create -f environment.yml
      conda activate operon-env

      # Use gcc-9 (or later)
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

3. To test that the python package installed correctly, try ``python -c "from operon.sklearn import SymbolicRegressor"``.

Windows/VcPkg
""""

Alternatively,  `vcpkg <https://vcpkg.io/en/index.html>`_ also works on Windows and Linux.

1. Install dependencies and clone the repo

    .. code-block:: Python
    
        # install dependencies
        vcpkg install ceres:x64-linux fmt:x64-linux pybind11:x64-linux cxxopts:x64-linux doctest:x64-linux python3:x64-linux taskflow:x64-linux
        
        # clone operon
        git clone https://github.com/heal-research/operon
        cd operon
        
        
2. Configure and build (make sure to use the appropriate generator for your system, e.g. ``-G "Visual Studio 16 2019" -A x64``. If the python path is not correctly detected, you can specify the install destination for the python module with ``-DCMAKE_INSTALL_PREFIX=<path>``

    .. code-block:: Python
    
        # configure
        mkdir build && cd build
        cmake .. -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON
        
        # build
        make -j pyoperon
        
3. To test that the python package installed correctly, try ``python -c "from operon.sklearn import SymbolicRegressor"``.

