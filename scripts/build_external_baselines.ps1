param(
  [ValidateSet("all", "nsg", "sptag", "diskann", "gate")]
  [string]$Baseline = "all",
  [string]$Configuration = "Release",
  [switch]$UseWsl
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ThirdParty = Join-Path $Root "third_party"

function ConvertTo-WslPath([string]$Path) {
  $resolved = (Resolve-Path $Path).Path
  $drive = $resolved.Substring(0, 1).ToLowerInvariant()
  $rest = $resolved.Substring(2).Replace("\", "/")
  return "/mnt/$drive$rest"
}

function Get-VsDevCmd {
  $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Build Tools with C++ workload."
  }
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vsPath) {
    throw "Visual Studio C++ build tools not found."
  }
  $devCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
  if (-not (Test-Path $devCmd)) {
    throw "VsDevCmd.bat not found under $vsPath"
  }
  return $devCmd
}

function Invoke-InVsDevShell([string]$Command) {
  $devCmd = Get-VsDevCmd
  cmd /c "`"$devCmd`" -arch=x64 -host_arch=x64 && $Command"
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE`: $Command"
  }
}

function Build-NSG {
  $repo = Join-Path $ThirdParty "nsg"
  $build = Join-Path $repo "build"
  New-Item -ItemType Directory -Force -Path $build | Out-Null
  Invoke-InVsDevShell "cmake -S `"$repo`" -B `"$build`" -DCMAKE_BUILD_TYPE=$Configuration"
  Invoke-InVsDevShell "cmake --build `"$build`" --config $Configuration --parallel"
}

function Build-NSG-WSL {
  $rootWsl = ConvertTo-WslPath $Root
  wsl bash -lc "set -euo pipefail; cd '$rootWsl/third_party/nsg'; rm -rf build; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE=$Configuration ..; make -j`$(nproc)"
  if ($LASTEXITCODE -ne 0) {
    throw "WSL NSG build failed"
  }
}

function Build-SPTAG {
  $repo = Join-Path $ThirdParty "SPTAG"
  $build = Join-Path $repo "build"
  New-Item -ItemType Directory -Force -Path $build | Out-Null
  Invoke-InVsDevShell "cmake -S `"$repo`" -B `"$build`" -A x64 -DSPDK=OFF -DROCKSDB=OFF"
  Invoke-InVsDevShell "cmake --build `"$build`" --config $Configuration --parallel"
}

function Build-SPTAG-WSL {
  $rootWsl = ConvertTo-WslPath $Root
  wsl bash -lc "set -euo pipefail; cd '$rootWsl/third_party/SPTAG'; rm -rf build; mkdir -p build; cd build; cmake -DSPDK=OFF -DROCKSDB=OFF -DTBB=ON -DCMAKE_BUILD_TYPE=$Configuration ..; make -j`$(nproc)"
  if ($LASTEXITCODE -ne 0) {
    throw "WSL SPTAG build failed"
  }
}

function Build-DiskANN {
  $repo = Join-Path $ThirdParty "DiskANN"
  if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo not found. DiskANN3 requires Rust/Cargo; install rustup, then rerun this target."
  }
  Push-Location $repo
  try {
    $env:RUSTFLAGS = "-Ctarget-cpu=x86-64-v3"
    cargo build --release --package diskann-benchmark
    if ($LASTEXITCODE -ne 0) {
      throw "cargo build failed"
    }
  } finally {
    Pop-Location
  }
}

function Build-DiskANN-WSL {
  $rootWsl = ConvertTo-WslPath $Root
  wsl bash -lc "set -euo pipefail; . `"`$HOME/.cargo/env`"; command -v cargo >/dev/null || { echo 'cargo not found. Install rustup in WSL for DiskANN3.' >&2; exit 127; }; cd '$rootWsl/third_party/DiskANN'; export RUSTFLAGS='-Ctarget-cpu=x86-64-v3'; cargo build --release --package diskann-benchmark"
  if ($LASTEXITCODE -ne 0) {
    throw "WSL DiskANN build failed"
  }
}

function Build-GATE {
  throw "GATE's artifact CMake uses GCC/Linux flags; run with -Baseline gate -UseWsl."
}

function Build-GATE-WSL {
  $rootWsl = ConvertTo-WslPath $Root
  wsl bash -lc "set -euo pipefail; cd '$rootWsl/third_party/GATE/jacksondca-gate-78334b4'; rm -rf build; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE=$Configuration ..; make -j`$(nproc)"
  if ($LASTEXITCODE -ne 0) {
    throw "WSL GATE build failed"
  }
}

if ($UseWsl) {
  if ($Baseline -eq "all" -or $Baseline -eq "nsg") { Build-NSG-WSL }
  if ($Baseline -eq "all" -or $Baseline -eq "sptag") { Build-SPTAG-WSL }
  if ($Baseline -eq "all" -or $Baseline -eq "diskann") { Build-DiskANN-WSL }
  if ($Baseline -eq "all" -or $Baseline -eq "gate") { Build-GATE-WSL }
} else {
  if ($Baseline -eq "all" -or $Baseline -eq "nsg") { Build-NSG }
  if ($Baseline -eq "all" -or $Baseline -eq "sptag") { Build-SPTAG }
  if ($Baseline -eq "all" -or $Baseline -eq "diskann") { Build-DiskANN }
  if ($Baseline -eq "all" -or $Baseline -eq "gate") { Build-GATE }
}
