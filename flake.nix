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

    # Last nixpkgs before ROCm bumped to LLVM 22; ACPP 25.x supports LLVM ≤ 20.
    # Used exclusively for adaptivecppWithRocm in devShells.rocm.
    nixpkgs-rocm.url = "github:nixos/nixpkgs/5f83595ff1ea7e10ae0e7d01cc233d19f6dd5ae1";

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
      nixpkgs-rocm,
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
            config.allowUnfree = true;
            overlays = [
              foolnotion.overlay
              (final: prev: {
                fluky = fluky.packages.${system}.default;
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
          # ACPP from the pinned nixpkgs (ROCm LLVM 20), overriding llvmPackages_18
          # → llvmPackages_20 so compiler and device-lib bitcode versions match.
          # ACPP 25.x supports LLVM ≤ 20; no experimental flag needed.
          pkgs-rocm = import inputs.nixpkgs-rocm {
            inherit system;
            config.allowUnfree = true;
          };
          adaptivecppWithRocm = (pkgs-rocm.adaptivecppWithRocm.override {
            llvmPackages_18 = pkgs-rocm.llvmPackages_20;
          }).overrideAttrs (old: {
            # Disable hiprtc: hiprtcLinkComplete(LLVM_BITCODE) is broken in ROCm 7.1.1.
            # Without HIPRTC_LIBRARY, ACPP uses clangJitLink (subprocess) instead.
            cmakeFlags = old.cmakeFlags ++ [ "-DHIPRTC_LIBRARY=" ];
            hardeningDisable = [ "all" ];
          });
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

          devShells.compare = pkgs.mkShell {
            name = "operon-compare";
            packages = [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  scipy
                  tabulate
                ]
              ))
            ];
          };

          # scipy + tabulate + ROCm JIT environment in one shell.
          # inputsFrom inherits buildInputs/nativeBuildInputs from .#rocm
          # (including adaptivecppWithRocm); shellHook must be repeated
          # because inputsFrom does not propagate it.
          devShells.rocm-compare = pkgs.mkShell {
            name = "operon-rocm-compare";
            inputsFrom = [ devShells.rocm ];
            packages = [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  scipy
                  tabulate
                ]
              ))
            ];
            shellHook = ''
              export NIX_HARDENING_ENABLE=""
              export ROCM_PATH=${adaptivecppWithRocm.rocmMerged}
              export PATH=${pkgs-rocm.rocmPackages."rocm-toolchain"}/bin:$PATH
            '';
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
              ++ pkgs.lib.optionals (pkgs.stdenv.hostPlatform.system == "x86_64-linux") (
                with pkgs;
                [
                  gdb
                  graphviz
                  hyperfine
                ]
              );
          };

          devShells.rocm = stdenv.mkDerivation {
            name = "operon-rocm";

            nativeBuildInputs =
              operon.nativeBuildInputs
              ++ (with pkgs; [
                clang-tools
                cmake-language-server
              ])
              ++ [ adaptivecppWithRocm ];

            buildInputs =
              operon.buildInputs
              ++ [ adaptivecppWithRocm ];

            # clangJitLink (ACPP's JIT path when hiprtc is disabled) calls the Nix
            # LLVM 20 clang wrapper as a subprocess to compile the SYCL kernel.
            # That wrapper needs:
            #  - NIX_HARDENING_ENABLE="" — strips flags unsupported for amdgcn
            #  - ROCM_PATH — lets clang find ROCm device libraries
            #  - rocm-toolchain/bin in PATH — provides lld and llvm-objcopy
            shellHook = ''
              export NIX_HARDENING_ENABLE=""
              export ROCM_PATH=${adaptivecppWithRocm.rocmMerged}
              export PATH=${pkgs-rocm.rocmPackages."rocm-toolchain"}/bin:$PATH
            '';
          };

          devShells.cuda = stdenv.mkDerivation {
            name = "operon-cuda";

            nativeBuildInputs =
              operon.nativeBuildInputs
              ++ (with pkgs; [
                adaptivecppWithCuda
                clang-tools
                cmake-language-server
              ]);

            buildInputs =
              operon.buildInputs
              ++ (with pkgs; [
                adaptivecppWithCuda
              ]);
          };

          apps.operon-gp.program = "${packages.default}/bin/operon_gp";
          apps.operon-nsgp.program = "${packages.default}/bin/operon_nsgp";
          apps.operon-parse-model.program = "${packages.default}/bin/operon_parse_model";
        };
    };
}
