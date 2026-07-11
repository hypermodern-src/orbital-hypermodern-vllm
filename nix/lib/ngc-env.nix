# NGC python environment helpers.
#
# Computes the library paths and LD_LIBRARY_PATH used by both the NixOS module
# and the interactive perSystem apps. The NGC python from the nvidia-sdk flake
# provides torch (cu130), triton, tensorrt_llm, etc. — all autoPatchelf'd.
{ lib, pkgs, ngcPython, ngcSdk }:
let
  ngcPkg = ngcPython.passthru.ngcPythonPackages;
  ngcSitePackages = "${ngcPkg}/${ngcPython.passthru.sitePackages}";
in
{
  inherit ngcPkg ngcSitePackages;

  ngcLib = "${ngcPkg}/lib";
  torchLib = "${ngcSitePackages}/torch/lib";
  trtLlmLibs = "${ngcSitePackages}/tensorrt_llm/libs";

  # Mirrors the LD_LIBRARY_PATH set by nvidia-sdk's python wrapper
  # (ngc-python.nix makeWrapper), plus libstdc++ from stdenv.cc.cc.lib.
  ldLibraryPath = lib.concatStringsSep ":" [
    "${pkgs.python312}/lib"
    "${ngcPkg}/lib"
    "${ngcSitePackages}/torch/lib"
    "${ngcSitePackages}/tensorrt_llm/libs"
    "${ngcSdk}/lib64"
    "${ngcSdk}/lib"
    "${pkgs.stdenv.cc.cc.lib}/lib"
    "/run/opengl-driver/lib"
  ];
}
