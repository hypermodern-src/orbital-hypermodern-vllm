{ inputs, ... }: {
  imports = [
    ./formatter.nix
    ./powercap.nix
    ./vllm.nix
    ./apps.nix
  ];
}
