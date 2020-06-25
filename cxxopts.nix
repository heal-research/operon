with import <nixos-unstable> {};

gcc10Stdenv.mkDerivation rec {
  pname = "cxxopts";
  version = "2.2.0";

  src = fetchFromGitHub {
    owner = "jarro2783";
    repo = "cxxopts";
    rev = "v${version}";
    sha256 = "1lz4v7jwp870ddrrks6kwh62c8hqc2pfdcpwshlmcf758li8ajz6";
  };

  buildinputs = [ gcc10Stdenv ];
  nativeBuildInputs = [ cmake ];

  meta = with gcc10Stdenv.lib; {
    description = "Lightweight C++ command line option parser";
    homepage = https://github.com/jarro2783/cxxopts;
    changelog = "";
    license = licenses.mit;
  };
}
