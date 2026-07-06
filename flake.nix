{
  description = "Operon development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    fluky.url = "github:foolnotion/fluky";
    lbfgs.url = "github:foolnotion/lbfgs";
    infix-parser.url = "github:foolnotion/infix-parser";
    ndsort.url = "github:foolnotion/ndsort";
    vstat.url = "github:heal-research/vstat";
    vdt.url = "github:foolnotion/vdt/master";
    pappus.url = "github:heal-research/pappus";

    # make everything follow nixpkgs
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    infix-parser.inputs.nixpkgs.follows = "nixpkgs";
    ndsort.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.foolnotion.follows = "foolnotion";
    vdt.inputs.nixpkgs.follows = "nixpkgs";
    fluky.inputs.nixpkgs.follows = "nixpkgs";
    pappus.inputs.nixpkgs.follows = "nixpkgs";
    pappus.inputs.foolnotion.follows = "foolnotion";
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
      pappus,
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
                glaze = prev.glaze.override { enableSSL = false; };
                # Unicode-aware --help wrapping isn't worth pulling in ICU
                # (and its static archives) just for line-wrapping cosmetics.
                cxxopts = prev.cxxopts.override { enableUnicodeHelp = false; };
                lbfgs = lbfgs.packages.${system}.default;
                infix-parser = infix-parser.packages.${system}.default;
                ndsort = ndsort.packages.${system}.default;
                vdt = vdt.packages.${system}.default;
                vstat = vstat.packages.${system}.default;
                pappus = pappus.packages.${system}.default;
              })
            ];
          };
          enableTesting = true;
          enableAsmjit = true;
          inherit (pkgs.llvmPackages_21) stdenv;
          mkOperon =
            { enableShared }:
            import ./operon.nix {
              inherit
                stdenv
                pkgs
                system
                enableShared
                enableTesting
                enableAsmjit
                ;
            };
          operonShared = mkOperon { enableShared = true; };
          operonStatic = mkOperon { enableShared = false; };
          operon = operonShared;
        in
        rec {
          packages = {
            default = operonShared.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            cli = operonShared.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            cli-static = operonStatic.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [
                "-DBUILD_CLI_PROGRAMS=ON"
                "-DCPM_USE_LOCAL_PACKAGES=ON"
              ];
            });

            library = operonShared;

            library-static = operonStatic;
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
              ++ operon.propagatedBuildInputs
              ++ (
                with pkgs;
                pkgs.lib.optionals (pkgs.stdenv.hostPlatform.system == "x86_64-linux") (
                  with pkgs;
                  [
                    gdb
                    graphviz
                    hyperfine
                    perf
                  ]
                )
              )
              ++ [
                # pappus: always required — IntervalEvaluator/AffineEvaluator
                # are public headers with an unconditional pappus dependency.
                pkgs.pappus
              ];
          };

          apps.operon-gp.program = "${packages.default}/bin/operon_gp";
          apps.operon-nsgp.program = "${packages.default}/bin/operon_nsgp";
          apps.operon-parse-model.program = "${packages.default}/bin/operon_parse_model";
        };
    };
}
