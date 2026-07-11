# Interactive perSystem apps: setup, serve, powercap, smoke, load, ocean.
{ inputs, lib, ... }:
{
  perSystem =
    { pkgs, system, ... }:
    let
      shared = import ../lib/vllm.nix { inherit lib; };
      ngcPython = inputs.nvidia-sdk.packages.${system}.python;
      ngcSdk = inputs.nvidia-sdk.packages.${system}.nvidia-sdk;
      ngcEnv = import ../lib/ngc-env.nix {
        inherit lib pkgs ngcPython ngcSdk;
      };

      # $VLLM_VENV overrides; default is ./virtualenvs/<name> relative to cwd.
      venvExpr = ''"''${VLLM_VENV:-$PWD/virtualenvs/${shared.venvName}}"'';

      runtimeDeps = with pkgs; [
        uv
        python3
        curl
        jq
        coreutils
        gnused
        gawk
      ];

      mkApp = pkg: {
        type = "app";
        program = lib.getExe pkg;
      };

      setup = pkgs.writeShellApplication {
        name = "vllm-setup";
        runtimeInputs = runtimeDeps ++ [ pkgs.python312 ];
        text = ''
          VENV=${venvExpr}
          echo ">>> creating venv from nixpkgs python3.12 at $VENV"
          uv venv --python ${pkgs.python312}/bin/python3 "$VENV"
          echo ">>> exposing NGC site-packages (torch, triton, …) via .pth file"
          echo "${ngcEnv.ngcSitePackages}" > "$VENV/lib/python3.12/site-packages/ngc.pth"
          echo ">>> installing vllm (torch/triton satisfied by NGC)"
          VIRTUAL_ENV="$VENV" uv pip install vllm
          echo ">>> matching flashinfer-jit-cache to installed flashinfer-python"
          FIVER=$(VIRTUAL_ENV="$VENV" uv pip show flashinfer-python | awk '/^Version:/{print $2}')
          echo "    flashinfer-python==$FIVER -> installing prebuilt jit-cache from ${shared.flashinferIndex}"
          VIRTUAL_ENV="$VENV" uv pip install --index-url ${shared.flashinferIndex} "flashinfer-jit-cache==$FIVER"
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
          export TRITON_LIBCUDA_PATH=${shared.tritonLibcuda}
          export CUDA_HOME="${ngcSdk}"
          export OPAL_PREFIX="${ngcEnv.ngcPkg}/ompi"
          export LD_LIBRARY_PATH="${ngcEnv.ldLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          # shellcheck disable=SC1091
          set +u; source "$VENV/bin/activate"; set -u
          # Any extra args ("$@") are appended, e.g.:  nix run .#serve -- --enforce-eager
          exec vllm serve ${shared.model} \
            ${shared.serveArgsShell} \
            "$@"
        '';
      };

      powercap = pkgs.writeShellApplication {
        name = "vllm-powercap";
        runtimeInputs = [ pkgs.coreutils ];
        text = ''
          LIMIT="''${1:-${toString shared.powerLimit}}"
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
            "model":"${shared.model}",
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
            "model": "${shared.model}",
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
            "model": "${shared.model}",
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
        default = serve;
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
          export TRITON_LIBCUDA_PATH=${shared.tritonLibcuda}
          export CUDA_HOME="${ngcSdk}"
          export OPAL_PREFIX="${ngcEnv.ngcPkg}/ompi"
          export LD_LIBRARY_PATH="${ngcEnv.ldLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
}
