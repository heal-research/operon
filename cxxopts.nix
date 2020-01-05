with import <nixpkgs> {};

stdenv.mkDerivation rec {
    pname = "cxxopts";
    version = "2.2.0";

    src = fetchurl {
        url = "https://github.com/jarro2783/cxxopts/archive/v2.2.0.tar.gz";
        sha256 = "447dbfc2361fce9742c5d1c9cfb25731c977b405f9085a738fbd608626da8a4d";
    };

    buildinputs = [ stdenv ];
    nativeBuildInputs = [ cmake ];

    meta = with stdenv.lib; {
        description = "Lightweight C++ command line option parser";
        homepage = https://github.com/jarro2783/cxxopts;
        changelog = "";
        license = licenses.mit;
    };
}
