{
  description = "// vllm // Qwen3.6-27B-NVFP4-MTP serving harness on Blackwell";

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem =
        { system, ... }:
        {
          _module.args.pkgs = import inputs.nixpkgs { inherit system; };
        };

      imports = [ ./nix/flake ];
    };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    # NVIDIA SDK — CUDA 13.1 + NGC 25.12 python (torch cu130, triton, tensorrt_llm)
    # extracted from the NGC Triton+TRT-LLM container. This replaces the
    # imperative uv venv: torch/triton/etc. come from the NGC extraction
    # (autoPatchelf'd, self-consistent) and only vllm + flashinfer-jit-cache
    # are pip-installed on top at runtime.
    nvidia-sdk = {
      url = "git+ssh://git@git.s4.gl/straylight/nvidia-sdk.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };
}
