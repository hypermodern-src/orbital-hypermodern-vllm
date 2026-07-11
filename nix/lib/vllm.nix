# Shared vLLM configuration for Qwen3.6-27B-NVFP4-MTP on Blackwell.
#
# This is the single source of truth used by both the NixOS module
# (nixosModules.vllm) and the interactive perSystem apps (serve, smoke, etc.).
{ lib }:
let
  model = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP";

  serveArgs = [
    "--trust-remote-code"
    "--quantization"
    "modelopt"
    "--language-model-only"
    "--max-model-len"
    "262144"
    "--max-num-seqs"
    "2"
    "--kv-cache-dtype"
    "fp8"
    "--gpu-memory-utilization"
    "0.9"
    "--reasoning-parser"
    "qwen3"
    "--enable-auto-tool-choice"
    "--tool-call-parser"
    "qwen3_xml"
    "--speculative-config"
    ''{"method":"qwen3_5_mtp","num_speculative_tokens":3}''
  ];
in
{
  inherit model serveArgs;

  venvName = "vllm-venv-qwen3.6-nvfp4-mtp";
  pythonVersion = "3.12";
  vllmVersion = "0.24.0";

  # flashinfer wheel index must match torch's CUDA (torch is +cu130).
  flashinferIndex = "https://flashinfer.ai/whl/cu130";

  # Triton locates libcuda by shelling out to `/sbin/ldconfig -p`, which does
  # not exist on NixOS; TRITON_LIBCUDA_PATH is triton's built-in escape hatch.
  tritonLibcuda = "/run/opengl-driver/lib";

  # Power limit in watts. 475 is the verified-stable ceiling on this RTX PRO
  # 6000 Blackwell; the 600W default triggers Xid 79 under sustained load.
  powerLimit = 475;

  # Default bind address/port for the managed service.
  host = "127.0.0.1";
  port = 8000;

  # Render the serve args as a single shell-escaped string for scripts that
  # build the command line manually.
  serveArgsShell = lib.concatStringsSep " " (map lib.escapeShellArg serveArgs);
}
