# NixOS module: run vLLM as a managed systemd service.
#
# The NGC 25.12 python environment (torch cu130, triton, tensorrt_llm, etc.)
# comes from the nvidia-sdk flake — autoPatchelf'd, self-consistent. A venv is
# created from nixpkgs python312 with a .pth file exposing the NGC
# site-packages, then only vllm + flashinfer-jit-cache are pip-installed on
# top in ExecStartPre.
{ inputs, lib, ... }:
let
  shared = import ../lib/vllm.nix { inherit lib; };
in
{
  flake.nixosModules.vllm =
    { config, pkgs, ... }:
    let
      cfg = config.services.vllm;
      venv = "${cfg.stateDir}/venv";

      ngcEnv = import ../lib/ngc-env.nix {
        inherit lib pkgs;
        ngcPython = cfg.ngcPython;
        ngcSdk = cfg.ngcSdk;
      };

      baseArgs = shared.serveArgs ++ [
        "--host" cfg.host
        "--port" (toString cfg.port)
      ];
      allArgs = baseArgs ++ cfg.extraArgs;

      bootstrap = pkgs.writeShellApplication {
        name = "vllm-bootstrap";
        runtimeInputs = [ pkgs.uv pkgs.coreutils pkgs.gawk pkgs.cacert pkgs.python312 ];
        text = ''
          WANT="vllm==${cfg.vllmVersion} ngc=${cfg.ngcPython.version}"
          if [ -x "${venv}/bin/vllm" ] && [ "$(cat "${venv}/.stamp" 2>/dev/null || true)" = "$WANT" ]; then
            echo "vllm venv already provisioned"; exit 0
          fi
          echo ">>> provisioning vllm venv at ${venv}"
          rm -rf "${venv}"
          # Venv from nixpkgs python312 (Nix binary — no nix-ld needed
          # for the interpreter itself, unlike the old uv-managed FHS python)
          uv venv --python ${pkgs.python312}/bin/python3 "${venv}"
          # Expose NGC site-packages (torch, triton, tensorrt_llm, …) to
          # the venv via a .pth file so pip sees them as installed.
          echo "${ngcEnv.ngcSitePackages}" > "${venv}/lib/python3.12/site-packages/ngc.pth"
          # Install vllm — torch/triton/etc. are satisfied by NGC.
          VIRTUAL_ENV="${venv}" uv pip install "vllm==${cfg.vllmVersion}"
          # Match flashinfer-jit-cache to the installed flashinfer-python
          FIVER=$(VIRTUAL_ENV="${venv}" uv pip show flashinfer-python | awk '/^Version:/{print $2}')
          echo ">>> flashinfer-python==$FIVER -> prebuilt jit-cache (${cfg.flashinferIndex})"
          VIRTUAL_ENV="${venv}" uv pip install --index-url ${cfg.flashinferIndex} "flashinfer-jit-cache==$FIVER"
          printf '%s' "$WANT" > "${venv}/.stamp"
        '';
      };

      start = pkgs.writeShellApplication {
        name = "vllm-start";
        runtimeInputs = [ ];
        text = ''
          exec "${venv}/bin/vllm" serve ${lib.escapeShellArg cfg.model} ${lib.escapeShellArgs allArgs}
        '';
      };
    in
    {
      imports = [ inputs.self.nixosModules.gpuPowercap ];

      options.services.vllm = {
        enable = lib.mkEnableOption "vLLM OpenAI-compatible server";
        model = lib.mkOption {
          type = lib.types.str;
          default = shared.model;
          description = "HuggingFace model id to serve.";
        };
        host = lib.mkOption {
          type = lib.types.str;
          default = shared.host;
          description = "Bind address.";
        };
        port = lib.mkOption {
          type = lib.types.port;
          default = shared.port;
        };
        user = lib.mkOption {
          type = lib.types.str;
          default = "vllm";
        };
        group = lib.mkOption {
          type = lib.types.str;
          default = "vllm";
        };
        stateDir = lib.mkOption {
          type = lib.types.path;
          default = "/var/lib/vllm";
          description = "Holds the venv, uv caches, and the HF model cache.";
        };
        ngcPython = lib.mkOption {
          type = lib.types.package;
          default = inputs.nvidia-sdk.packages.${pkgs.system}.python;
          description = "NGC-extracted Python environment from the nvidia-sdk flake. Provides torch (cu130), triton, tensorrt_llm, etc. — all self-consistent and autoPatchelf'd.";
        };
        ngcSdk = lib.mkOption {
          type = lib.types.package;
          default = inputs.nvidia-sdk.packages.${pkgs.system}.nvidia-sdk;
          description = "NVIDIA SDK package (CUDA, cuDNN, NCCL, TensorRT, etc. from the nvidia-sdk flake).";
        };
        vllmVersion = lib.mkOption {
          type = lib.types.str;
          default = shared.vllmVersion;
          description = "vllm wheel version pip-installed on top of the NGC python.";
        };
        flashinferIndex = lib.mkOption {
          type = lib.types.str;
          default = shared.flashinferIndex;
          description = "flashinfer-jit-cache wheel index (must match torch CUDA).";
        };
        extraArgs = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [ ];
          example = [ "--enforce-eager" ];
          description = "Extra flags appended to `vllm serve` (later flags win).";
        };
        environmentFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = "EnvironmentFile for secrets like HF_TOKEN.";
        };
        openFirewall = lib.mkOption {
          type = lib.types.bool;
          default = false;
        };
        powercap = {
          enable = lib.mkOption {
            type = lib.types.bool;
            default = true;
            description = "Also apply the 475W GPU power cap (strongly recommended).";
          };
          watts = lib.mkOption {
            type = lib.types.int;
            default = shared.powerLimit;
          };
        };
      };

      config = lib.mkIf cfg.enable {
        assertions = [
          {
            assertion = config.programs.nix-ld.enable;
            message = "services.vllm requires programs.nix-ld.enable = true (nix-ld provides libstdc++ fallback for vllm wheel .so files).";
          }
        ];

        services.gpuPowercap = lib.mkIf cfg.powercap.enable {
          enable = true;
          watts = cfg.powercap.watts;
        };

        users.users.${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = cfg.stateDir;
          description = "vLLM service user";
        };
        users.groups.${cfg.group} = { };

        networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

        systemd.tmpfiles.rules = [
          "d ${cfg.stateDir} 0750 ${cfg.user} ${cfg.group} - -"
        ];

        systemd.services.vllm = {
          description = "vLLM OpenAI server (${cfg.model})";
          wantedBy = [ "multi-user.target" ];
          wants = [ "network-online.target" ];
          after =
            [ "network-online.target" "nvidia-persistenced.service" ]
            ++ lib.optional cfg.powercap.enable "gpu-powercap.service";

          environment = {
            HOME = cfg.stateDir;
            XDG_CACHE_HOME = "${cfg.stateDir}/cache";
            UV_CACHE_DIR = "${cfg.stateDir}/uv/cache";
            UV_PYTHON_INSTALL_DIR = "${cfg.stateDir}/uv/python";
            HF_HOME = "${cfg.stateDir}/huggingface";
            # --- NGC runtime env (mirrors nvidia-sdk python wrapper) ---
            TRITON_LIBCUDA_PATH = shared.tritonLibcuda;
            CUDA_HOME = "${cfg.ngcSdk}";
            OPAL_PREFIX = "${ngcEnv.ngcPkg}/ompi";
            LD_LIBRARY_PATH = ngcEnv.ldLibraryPath;
            # --- nix-ld fallback for any FHS binaries in the vllm wheel ---
            NIX_LD = "/run/current-system/sw/share/nix-ld/lib/ld.so";
            NIX_LD_LIBRARY_PATH = "/run/current-system/sw/share/nix-ld/lib";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          };

          serviceConfig = {
            Type = "simple";
            User = cfg.user;
            Group = cfg.group;
            WorkingDirectory = cfg.stateDir;
            ExecStartPre = "${lib.getExe bootstrap}";
            ExecStart = "${lib.getExe start}";
            Restart = "on-failure";
            RestartSec = 15;
            # first boot provisions the venv (pip) + downloads ~18GB model
            TimeoutStartSec = "3600";
            EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
          };
        };
      };
    };

  flake.nixosModules.default = inputs.self.nixosModules.vllm;
}
