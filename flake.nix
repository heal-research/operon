{
  description = "Operon development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    fluky.url = "github:foolnotion/fluky";
    lbfgs.url = "github:foolnotion/lbfgs";
    infix-parser.url = "github:foolnotion/infix-parser";
    ndsort.url = "github://git@github.com/foolnotion/ndsort";
    vstat.url = "github:heal-research/vstat";
    vdt.url = "github:foolnotion/vdt/master";

    # make everything follow nixpkgs
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    infix-parser.inputs.nixpkgs.follows = "nixpkgs";
    ndsort.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.foolnotion.follows = "foolnotion";
    vdt.inputs.nixpkgs.follows = "nixpkgs";
    fluky.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      foolnotion,
      fluky,
      infix-parser,
      ndsort,
      vdt,
      vstat,
      lbfgs,
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, system, ... }:
        let
          pkgs = import self.inputs.nixpkgs {
            inherit system;
            overlays = [
              foolnotion.overlay
              (final: prev: {
                fluky = fluky.packages.${system}.default;
                glaze = prev.glaze.overrideAttrs (old: {
                  cmakeFlags = (old.cmakeFlags or []) ++ [ "-DGLAZE_ENABLE_SSL=OFF" ];
                });
                lbfgs = lbfgs.packages.${system}.default;
                infix-parser = infix-parser.packages.${system}.default;
                ndsort = ndsort.packages.${system}.default;
                vdt = vdt.packages.${system}.default;
                vstat = vstat.packages.${system}.default;
              })
            ];
          };
          enableShared = !pkgs.stdenv.hostPlatform.isStatic;
          enableTesting = true;
          inherit (pkgs.llvmPackages_21) stdenv;
          operon = import ./operon.nix {
            inherit stdenv pkgs system;
            inherit enableShared enableTesting;
          };
        in
        rec {
          packages = {
            default = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            cli = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            cli-static = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            library = operon.overrideAttrs (old: {
              enableShared = true;
            });

            library-static = operon.overrideAttrs (old: {
              enableShared = false;
            });
          };

          devShells.tools = pkgs.mkShell {
            name = "operon-tools";
            packages = [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  duckdb
                  pandas
                  pyarrow
                  optuna
                  rich
                  loguru
                  scipy
                  pyyaml
                  pytest
                  tabulate
                  matplotlib
                  numpy
                ]
              ))
            ];
          };

          devShells.default = stdenv.mkDerivation {
            name = "operon";

            nativeBuildInputs =
              operon.nativeBuildInputs
              ++ (with pkgs; [
                clang-tools
                cppcheck
                include-what-you-use
                cmake-language-server
              ]);

            buildInputs =
              operon.buildInputs
              ++ [ pkgs.asmjit ]
              ++ pkgs.lib.optionals (pkgs.stdenv.hostPlatform.system == "x86_64-linux") (
                with pkgs;
                [
                  gdb
                  graphviz
                  hyperfine
                  linuxPackages.perf
                ]
              );
          };

          apps.operon-gp.program = "${packages.default}/bin/operon_gp";
          apps.operon-nsgp.program = "${packages.default}/bin/operon_nsgp";
          apps.operon-parse-model.program = "${packages.default}/bin/operon_parse_model";
        };
    };
}
