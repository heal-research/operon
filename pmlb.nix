{ lib, pythonPackages, buildPythonPackage, fetchPypi }:

buildPythonPackage rec {
    pname = "pmlb";
    version = "1.0.2a0";

    src = fetchPypi {
      inherit pname version;

      sha256 = "1awlhyrkvdgf9br71dq53h5pn9yqhiw2lsa8ydknjp9rakimwsrl";
    };

    buildInputs = with pythonPackages; [ pandas requests pyyaml scikitlearn ];

    doCheck = false;

    meta = with lib; {
      homepage = "https://github.com/EpistasisLab/penn-ml-benchmarks";
      description = "A Python wrapper for the Penn Machine Learning Benchmark data repository.";
      license = licenses.mit;
    };
}
