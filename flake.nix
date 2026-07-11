{
  description = "vLLM serving harness for Qwen3.6-27B-NVFP4-MTP on Blackwell (SM120, RTX PRO 6000)";

  # ---------------------------------------------------------------------------
  # War-story notes (why every knob below exists), so we never re-derive them:
  #
  #  * NGC python from the nvidia-sdk flake provides torch (cu130), triton,
  #    tensorrt_llm, etc. — all self-consistent and autoPatchelf'd. A venv is
  #    created from nixpkgs python312 with a .pth file exposing the NGC
  #    site-packages, then only vllm + flashinfer-jit-cache are pip-installed
  #    on top. This avoids both nixpkgs CUDA rebuilds and the uv-managed FHS
  #    python that needed nix-ld for its interpreter.
  #
  #  * export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib. Triton locates
  #    libcuda by shelling out to `/sbin/ldconfig -p`, which does not exist
  #    on NixOS; TRITON_LIBCUDA_PATH is triton's built-in escape hatch
  #    (triton/knobs.py: TRITON_LIBCUDA_PATH). /run/opengl-driver/lib always
  #    holds the libcuda that matches the running kernel module.
  #
  #  * flashinfer-jit-cache (prebuilt SM120 kernels) so FlashInfer does NOT need
  #    a CUDA toolkit / nvcc at runtime to build the CUTLASS NVFP4 GEMM. Version
  #    MUST match the installed flashinfer-python; index is per-CUDA (cu130).
  #
  #  * --tool-call-parser qwen3_xml (NOT hermes). Qwen3.x emits XML tool calls
  #    (<function=...><parameter=...>); hermes silently leaves them in content.
  #
  #  * Power cap 475W. At the stock 600W cap this card threw Xid 79 (GPU fell
  #    off the bus) under sustained load and hard-locked the host. Capped to
  #    475W it ran sustained, clean. See services.gpuPowercap NixOS module.
  # ---------------------------------------------------------------------------

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
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      flake = {
        # -------------------------------------------------------------------
        # nixosModules.gpuPowercap — persistence + power cap at boot.
        # -------------------------------------------------------------------
        nixosModules.gpuPowercap =
          { config, lib, ... }:
          let
            cfg = config.services.gpuPowercap;
          in
          {
            options.services.gpuPowercap = {
              enable = lib.mkEnableOption "NVIDIA GPU power cap + persistence";
              watts = lib.mkOption {
                type = lib.types.int;
                default = 475;
                description = "Power limit in watts. 475 is the verified-stable ceiling on this RTX PRO 6000 Blackwell; the 600W default triggers Xid 79 under sustained load.";
              };
            };
            config = lib.mkIf cfg.enable {
              hardware.nvidia.nvidiaPersistenced = lib.mkDefault true;
              systemd.services.gpu-powercap = {
                description = "Cap NVIDIA GPU power limit to ${toString cfg.watts}W";
                wantedBy = [ "multi-user.target" ];
                after = [ "nvidia-persistenced.service" ];
                wants = [ "nvidia-persistenced.service" ];
                serviceConfig = {
                  Type = "oneshot";
                  RemainAfterExit = true;
                  ExecStart = "${lib.getExe' config.hardware.nvidia.package "nvidia-smi"} -pl ${toString cfg.watts}";
                };
              };
            };
          };

        # -------------------------------------------------------------------
        # nixosModules.vllm — run vLLM as a managed systemd service.
        #
        # The NGC 25.12 python environment (torch cu130, triton, tensorrt_llm,
        # etc.) comes from the nvidia-sdk flake — autoPatchelf'd, self-consistent,
        # no nixpkgs CUDA. A venv is created from nixpkgs python312 with a .pth
        # file exposing the NGC site-packages, then vllm + flashinfer-jit-cache
        # are pip-installed on top in ExecStartPre (the one impure seam — swap
        # for an FOD when ready). The nix-ld / triton / flashinfer / power-cap
        # knowledge is fully declarative here.
        # -------------------------------------------------------------------
        nixosModules.vllm =
          { config, lib, pkgs, ... }:
          let
            cfg = config.services.vllm;
            venv = "${cfg.stateDir}/venv";

            # NGC python environment from nvidia-sdk flake — provides torch
            # (cu130), triton, tensorrt_llm, etc. all self-consistent and
            # autoPatchelf'd. We create a venv from nixpkgs python312 and
            # expose the NGC site-packages via a .pth file, then pip-install
            # only vllm + flashinfer-jit-cache on top.
            ngcPkg = cfg.ngcPython.passthru.ngcPythonPackages;
            ngcSitePackages = "${ngcPkg}/${cfg.ngcPython.passthru.sitePackages}";
            ngcLib = "${ngcPkg}/lib";
            torchLib = "${ngcSitePackages}/torch/lib";
            trtLlmLibs = "${ngcSitePackages}/tensorrt_llm/libs";
            sdkLib64 = "${cfg.ngcSdk}/lib64";
            sdkLib = "${cfg.ngcSdk}/lib";

            # Mirrors the LD_LIBRARY_PATH set by nvidia-sdk's python wrapper
            # (ngc-python.nix makeWrapper). The vllm wheel's .so files find
            # torch/cuda/etc. via this; libstdc++ from stdenv.cc.cc.lib.
            ldLibraryPath = lib.concatStringsSep ":" [
              "${pkgs.python312}/lib"
              ngcLib
              torchLib
              trtLlmLibs
              sdkLib64
              sdkLib
              "${pkgs.stdenv.cc.cc.lib}/lib"
              "/run/opengl-driver/lib"
            ];

            baseArgs = [
              "--trust-remote-code"
              "--quantization" "modelopt"
              "--language-model-only"
              "--max-model-len" "262144"
              "--max-num-seqs" "2"
              "--kv-cache-dtype" "fp8"
              "--gpu-memory-utilization" "0.9"
              "--reasoning-parser" "qwen3"
              "--enable-auto-tool-choice"
              "--tool-call-parser" "qwen3_xml"
              "--speculative-config" ''{"method":"qwen3_5_mtp","num_speculative_tokens":3}''
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
                echo "${ngcSitePackages}" > "${venv}/lib/python3.12/site-packages/ngc.pth"
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
                default = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP";
                description = "HuggingFace model id to serve.";
              };
              host = lib.mkOption {
                type = lib.types.str;
                default = "127.0.0.1";
                description = "Bind address.";
              };
              port = lib.mkOption {
                type = lib.types.port;
                default = 8000;
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
                default = "0.24.0";
                description = "vllm wheel version pip-installed on top of the NGC python.";
              };
              flashinferIndex = lib.mkOption {
                type = lib.types.str;
                default = "https://flashinfer.ai/whl/cu130";
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
                  default = 475;
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
                  TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
                  CUDA_HOME = "${cfg.ngcSdk}";
                  OPAL_PREFIX = "${ngcPkg}/ompi";
                  LD_LIBRARY_PATH = ldLibraryPath;
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

        nixosModules.default = inputs.self.nixosModules.vllm;
      };

      perSystem =
        { pkgs, lib, system, ... }:
        let
          ngcPython = inputs.nvidia-sdk.packages.${system}.python;
          ngcSdk = inputs.nvidia-sdk.packages.${system}.nvidia-sdk;
          ngcPkg = ngcPython.passthru.ngcPythonPackages;
          ngcSitePackages = "${ngcPkg}/${ngcPython.passthru.sitePackages}";

          # Mirrors the nvidia-sdk python wrapper's LD_LIBRARY_PATH.
          ngcLdLibraryPath = lib.concatStringsSep ":" [
            "${pkgs.python312}/lib"
            "${ngcPkg}/lib"
            "${ngcSitePackages}/torch/lib"
            "${ngcSitePackages}/tensorrt_llm/libs"
            "${ngcSdk}/lib64"
            "${ngcSdk}/lib"
            "${pkgs.stdenv.cc.cc.lib}/lib"
            "/run/opengl-driver/lib"
          ];

          cfg = {
            model = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP";
            venvName = "vllm-venv-qwen3.6-nvfp4-mtp";
            # flashinfer wheel index must match torch's CUDA (torch is +cu130).
            flashinferIndex = "https://flashinfer.ai/whl/cu130";
            tritonLibcuda = "/run/opengl-driver/lib";
            powerLimit = 475;
          };

          # $VLLM_VENV overrides; default is ./virtualenvs/<name> relative to cwd.
          venvExpr = ''"''${VLLM_VENV:-$PWD/virtualenvs/${cfg.venvName}}"'';

          runtimeDeps = with pkgs; [
            uv
            python3
            curl
            jq
            coreutils
            gnused
            gawk
          ];

          setup = pkgs.writeShellApplication {
            name = "vllm-setup";
            runtimeInputs = runtimeDeps ++ [ pkgs.python312 ];
            text = ''
              VENV=${venvExpr}
              echo ">>> creating venv from nixpkgs python3.12 at $VENV"
              uv venv --python ${pkgs.python312}/bin/python3 "$VENV"
              echo ">>> exposing NGC site-packages (torch, triton, …) via .pth file"
              echo "${ngcSitePackages}" > "$VENV/lib/python3.12/site-packages/ngc.pth"
              echo ">>> installing vllm (torch/triton satisfied by NGC)"
              VIRTUAL_ENV="$VENV" uv pip install vllm
              echo ">>> matching flashinfer-jit-cache to installed flashinfer-python"
              FIVER=$(VIRTUAL_ENV="$VENV" uv pip show flashinfer-python | awk '/^Version:/{print $2}')
              echo "    flashinfer-python==$FIVER -> installing prebuilt jit-cache from ${cfg.flashinferIndex}"
              VIRTUAL_ENV="$VENV" uv pip install --index-url ${cfg.flashinferIndex} "flashinfer-jit-cache==$FIVER"
              echo ">>> done. Serve with:  nix run .#serve"
            '';
          };

          serve = pkgs.writeShellApplication {
            name = "vllm-serve";
            runtimeInputs = runtimeDeps;
            text = ''
              VENV=${venvExpr}
              if [ ! -x "$VENV/bin/vllm" ]; then
                echo "!! venv not found at $VENV — run 'nix run .#setup' first" >&2
                exit 1
              fi
              # NGC runtime env (mirrors nvidia-sdk python wrapper + flake header)
              export TRITON_LIBCUDA_PATH=${cfg.tritonLibcuda}
              export CUDA_HOME="${ngcSdk}"
              export OPAL_PREFIX="${ngcPkg}/ompi"
              export LD_LIBRARY_PATH="${ngcLdLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              # shellcheck disable=SC1091
              set +u; source "$VENV/bin/activate"; set -u
              # Any extra args ("$@") are appended, e.g.:  nix run .#serve -- --enforce-eager
              exec vllm serve ${cfg.model} \
                --trust-remote-code \
                --quantization modelopt \
                --language-model-only \
                --max-model-len 262144 \
                --max-num-seqs 2 \
                --kv-cache-dtype fp8 \
                --gpu-memory-utilization 0.9 \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice \
                --tool-call-parser qwen3_xml \
                --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
                "$@"
            '';
          };

          powercap = pkgs.writeShellApplication {
            name = "vllm-powercap";
            runtimeInputs = [ pkgs.coreutils ];
            text = ''
              LIMIT="''${1:-${toString cfg.powerLimit}}"
              echo ">>> persistence on + power limit ''${LIMIT}W (needs sudo)"
              sudo nvidia-smi -pm 1
              sudo nvidia-smi -pl "$LIMIT"
              nvidia-smi --query-gpu=persistence_mode,power.limit,power.max_limit --format=csv
            '';
          };

          smoke = pkgs.writeShellApplication {
            name = "vllm-smoke";
            runtimeInputs = runtimeDeps;
            text = ''
              URL="''${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
              echo ">>> tool-call probe"
              curl -s "$URL" -H 'Content-Type: application/json' -d '{
                "model":"${cfg.model}",
                "messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],
                "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
                "tool_choice":"auto","max_tokens":256,"temperature":0,
                "chat_template_kwargs":{"enable_thinking":false}
              }' | jq '{finish:.choices[0].finish_reason, tool_calls:.choices[0].message.tool_calls}'
            '';
          };

          load = pkgs.writeShellApplication {
            name = "vllm-load";
            runtimeInputs = runtimeDeps;
            text = ''
              DURATION="''${1:-900}"
              URL="''${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
              echo ">>> sustained load for ''${DURATION}s (2 concurrent workers)"
              FILLER=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*1200)")
              BODY=$(python3 - "$FILLER" <<'PY'
              import json, sys
              print(json.dumps({
                "model": "${cfg.model}",
                "temperature": 0.7, "max_tokens": 1200,
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{"role": "user", "content":
                  "Context:\n" + sys.argv[1] +
                  "\n\nWrite a long, detailed technical essay on GPU memory hierarchies, HBM, caches, and coalescing. Do not stop early."}],
              }))
              PY
              )
              END=$(( $(date +%s) + DURATION ))
              worker() {
                local id="$1" n=0
                while [ "$(date +%s)" -lt "$END" ]; do
                  n=$((n + 1))
                  code=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY" -o /dev/null -w '%{http_code} %{time_total}s')
                  echo "worker$id req$n: $code"
                done
              }
              worker 1 &
              worker 2 &
              wait
              echo ">>> load complete"
            '';
          };

          ocean = pkgs.writeShellApplication {
            name = "vllm-ocean";
            runtimeInputs = runtimeDeps;
            text = ''
              PROMPT="''${1:-Write one long, continuous, unbroken stream of vivid prose about the ocean. Wave after wave, no lists, no headings. Do not stop early.}"
              MAXTOK="''${2:-2048}"
              TEMP="''${3:-0}"
              URL="''${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
              curl -N -s "$URL" -H 'Content-Type: application/json' \
                -d "$(python3 -c '
              import json, sys
              print(json.dumps({
                "model": "${cfg.model}",
                "stream": True, "stream_options": {"include_usage": True},
                "temperature": float(sys.argv[3]), "max_tokens": int(sys.argv[2]),
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{"role": "user", "content": sys.argv[1]}],
              }))' "$PROMPT" "$MAXTOK" "$TEMP")" \
              | python3 -c '
              import sys, json, time
              t0=None; comp=0; phase=None
              def hdr(l):
                  sys.stdout.write("\n\033[1;36m========== %s ==========\033[0m\n" % l); sys.stdout.flush()
              for line in sys.stdin:
                  line=line.strip()
                  if not line.startswith("data:"): continue
                  data=line[5:].strip()
                  if data=="[DONE]": break
                  try: o=json.loads(data)
                  except Exception: continue
                  ch=o.get("choices") or []
                  if ch:
                      d=ch[0].get("delta",{}) or {}
                      r=d.get("reasoning_content") or d.get("reasoning"); c=d.get("content")
                      if r:
                          if t0 is None: t0=time.time()
                          if phase!="think": hdr("THINKING"); phase="think"
                          sys.stdout.write(r); sys.stdout.flush()
                      if c:
                          if t0 is None: t0=time.time()
                          if phase!="answer": hdr("ANSWER"); phase="answer"
                          sys.stdout.write(c); sys.stdout.flush()
                  u=o.get("usage")
                  if u: comp=u.get("completion_tokens",0)
              dt=time.time()-t0 if t0 else 0
              print("\n\n\033[1;33m--- %d tokens | %.2fs | %.1f tok/s ---\033[0m" % (comp, dt, (comp/dt if dt else 0)))
              '
            '';
          };

          mkApp = pkg: {
            type = "app";
            program = lib.getExe pkg;
          };
        in
        {
          packages = {
            inherit
              setup
              serve
              powercap
              smoke
              load
              ocean
              ;
          };

          apps = {
            setup = mkApp setup;
            serve = mkApp serve;
            powercap = mkApp powercap;
            smoke = mkApp smoke;
            load = mkApp load;
            ocean = mkApp ocean;
            default = mkApp serve;
          };

          devShells.default = pkgs.mkShell {
            packages = runtimeDeps ++ [
              setup
              serve
              powercap
              smoke
              load
              ocean
            ];
            shellHook = ''
              export TRITON_LIBCUDA_PATH=${cfg.tritonLibcuda}
              export CUDA_HOME="${ngcSdk}"
              export OPAL_PREFIX="${ngcPkg}/ompi"
              export LD_LIBRARY_PATH="${ngcLdLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              cat <<'EOF'
              vLLM Qwen3.6-27B-NVFP4-MTP harness (NGC python + nixpkgs venv)
                nix run .#setup      create the venv + install vllm & flashinfer-jit-cache
                nix run .#powercap   persistence + 475W cap (sudo)
                nix run .#serve      launch the server (append flags after --)
                nix run .#smoke      tool-call sanity probe
                nix run .#ocean      streaming think+answer demo w/ tok/s
                nix run .#load [sec] sustained 2-worker load test
              EOF
            '';
          };
        };
    };
}
