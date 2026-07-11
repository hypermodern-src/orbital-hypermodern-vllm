# NixOS module: GPU persistence + power cap at boot.
{ lib, ... }:
{
  flake.nixosModules.gpuPowercap =
    { config, ... }:
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
}
