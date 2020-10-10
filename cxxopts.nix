with import <nixos-unstable> {};

gcc10Stdenv.mkDerivation rec {
  pname = "cxxopts";
  version = "2.2.1";

  stdenv = pkgs.gcc10Stdenv;

  src = fetchFromGitHub {
    owner = "jarro2783";
    repo = "cxxopts";
    rev = "v${version}";
    sha256 = "0d3y747lsh1wkalc39nxd088rbypxigm991lk3j91zpn56whrpha";
  };

  nativeBuildInputs = [ cmake ];

  meta = with pkgs.gcc10Stdenv.lib; {
    description = "Lightweight C++ command line option parser";
    homepage = https://github.com/jarro2783/cxxopts;
    changelog = "";
    license = licenses.mit;
  };
}
