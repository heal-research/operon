{
  description = "Operon development environment";
  nixConfig.bash-prompt = "\\[\\e[93m\\e[1m\\][operon-dev:\\[\\e[92m\\e[1m\\]\\w]$\\[\\e[0m\\] ";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, nur }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nur.overlay ];
          };
        in
        {
          devShell = pkgs.gcc11Stdenv.mkDerivation {
            name = "operon-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ cmake clang_12 clang-tools ];
            buildInputs = with pkgs; [
                # python environment for bindings and scripting
                (python39.withPackages (ps: with ps; [ pybind11 ]))
                # Project dependencies and utils for profiling and debugging
                pkgs.nur.repos.foolnotion.taskflow
                ceres-solver
                cmake
                cxxopts
                diff-so-fancy
                doctest
                eigen
                fmt
                gdb
                glog
                hyperfine
                jemalloc
                linuxPackages.perf
                mimalloc
                ninja
                openlibm
                valgrind
              ];

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
