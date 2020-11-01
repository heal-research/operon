{ lib, pythonPackages, buildPythonPackage, fetchPypi }:

buildPythonPackage rec {
    pname = "eli5";
    version = "0.10.1";

    src = fetchPypi {
      inherit pname version;

      sha256 = "0arapv99q8hs1ibcvdvfz1sx5yza8hhi2i01hfg2kwszlpbbbx32";
    };

    buildInputs = with pythonPackages; [ attrs scikitlearn graphviz six tabulate jinja2 ];

    doCheck = false;

    meta = with lib; {
      homepage = "https://eli5.readthedocs.io";
      description = "Python library which allows to visualize and debug various Machine Learning models using unified API.";
      license = licenses.mit;
    };
}
