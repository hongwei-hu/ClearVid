<#
.SYNOPSIS
    Build ClearVid.exe launcher using PyInstaller.

.DESCRIPTION
    Creates a lightweight ClearVid.exe (~15-25 MB) that:
      1. First run: detects GPU, downloads PyTorch/PySide6/etc. into lib/
      2. Subsequent runs: launches the GUI directly

    The .exe only bundles Python stdlib + ClearVid bootstrap code.
    Heavy dependencies (PyTorch ~2.5GB, PySide6, opencv) are installed at first run.

.PARAMETER OutputDir
    Directory where the build output goes. Default: dist/

.PARAMETER SkipInstallPyInstaller
    Skip installing PyInstaller (if already installed).

.EXAMPLE
    .\build_exe.ps1
    .\build_exe.ps1 -OutputDir D:\builds
#>

param(
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "dist"),
    [switch]$SkipInstallPyInstaller
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ClearVid EXE Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1: Ensure PyInstaller is available
# ---------------------------------------------------------------------------
if (-not $SkipInstallPyInstaller) {
    Write-Host "[1/4] Checking PyInstaller..." -ForegroundColor Yellow
    try {
        python -m PyInstaller --version 2>&1 | Out-Null
        Write-Host "       PyInstaller already installed" -ForegroundColor Gray
    } catch {
        Write-Host "       Installing PyInstaller..." -ForegroundColor Gray
        python -m pip install pyinstaller --quiet
    }
}

# ---------------------------------------------------------------------------
# Step 2: Run PyInstaller
# ---------------------------------------------------------------------------
Write-Host "[2/4] Building ClearVid.exe..." -ForegroundColor Yellow

$specFile = Join-Path $repoRoot "scripts\ClearVid.spec"
$distTarget = $OutputDir

Push-Location $repoRoot
try {
    python -m PyInstaller $specFile `
        --noconfirm `
        --distpath $distTarget `
        --workpath (Join-Path $OutputDir "build_temp")
} finally {
    Pop-Location
}

$exeDir = Join-Path $distTarget "ClearVid"
$exePath = Join-Path $exeDir "ClearVid.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "ERROR: ClearVid.exe was not created!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Step 3: Copy runtime files alongside the exe
# ---------------------------------------------------------------------------
Write-Host "[3/4] Copying runtime files..." -ForegroundColor Yellow

# FFmpeg
$ffmpegSrc = Join-Path $repoRoot "ffmpeg.exe"
if (Test-Path $ffmpegSrc) {
    Copy-Item -Path $ffmpegSrc -Destination (Join-Path $exeDir "ffmpeg.exe") -Force
    Write-Host "       ffmpeg.exe (from repo root)" -ForegroundColor Gray
} else {
    $sysFFmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($sysFFmpeg) {
        Copy-Item -Path $sysFFmpeg.Source -Destination (Join-Path $exeDir "ffmpeg.exe") -Force
        Write-Host "       ffmpeg.exe (from PATH)" -ForegroundColor Gray
    } else {
        Write-Host "       WARNING: ffmpeg.exe not found!" -ForegroundColor Red
    }
}

$ffprobeSrc = Join-Path $repoRoot "ffprobe.exe"
if (Test-Path $ffprobeSrc) {
    Copy-Item -Path $ffprobeSrc -Destination (Join-Path $exeDir "ffprobe.exe") -Force
    Write-Host "       ffprobe.exe (from repo root)" -ForegroundColor Gray
} else {
    $sysFFprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($sysFFprobe) {
        Copy-Item -Path $sysFFprobe.Source -Destination (Join-Path $exeDir "ffprobe.exe") -Force
        Write-Host "       ffprobe.exe (from PATH)" -ForegroundColor Gray
    } else {
        Write-Host "       WARNING: ffprobe.exe not found!" -ForegroundColor Red
    }
}

# Vendor (basicsr)
$vendorSrc = Join-Path $repoRoot "vendor"
if (Test-Path $vendorSrc) {
    $vendorDst = Join-Path $exeDir "vendor"
    if (-not (Test-Path $vendorDst)) {
        Copy-Item -Path $vendorSrc -Destination $vendorDst -Recurse -Force
        Get-ChildItem -Path $vendorDst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
        Write-Host "       vendor/ (basicsr)" -ForegroundColor Gray
    }
}

# Presets
$presetsSrc = Join-Path $repoRoot "clearvid\app\presets"
$presetsDst = Join-Path $exeDir "clearvid\app\presets"
if ((Test-Path $presetsSrc) -and -not (Test-Path $presetsDst)) {
    New-Item -ItemType Directory -Force -Path $presetsDst | Out-Null
    Copy-Item -Path "$presetsSrc\*" -Destination $presetsDst -Force
}

# Create placeholder directories
foreach ($dir in @("outputs", "weights", "samples")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $exeDir $dir) | Out-Null
}

# Samples
$samplesSrc = Join-Path $repoRoot "samples"
if (Test-Path $samplesSrc) {
    Copy-Item -Path "$samplesSrc\*" -Destination (Join-Path $exeDir "samples") -Force -ErrorAction SilentlyContinue
}

# Weight README
$weightReadme = Join-Path $repoRoot "weights\README.zh-CN.md"
if (Test-Path $weightReadme) {
    Copy-Item -Path $weightReadme -Destination (Join-Path $exeDir "weights\README.zh-CN.md") -Force
}

# ---------------------------------------------------------------------------
# Step 4: Summary
# ---------------------------------------------------------------------------
Write-Host "[4/4] Cleaning up..." -ForegroundColor Yellow
$buildTemp = Join-Path $OutputDir "build_temp"
if (Test-Path $buildTemp) {
    Remove-Item -Recurse -Force $buildTemp
}

$exeSize = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
$totalSize = [math]::Round(((Get-ChildItem -Path $exeDir -Recurse | Measure-Object -Property Length -Sum).Sum) / 1MB, 1)

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  EXE:   $exePath" -ForegroundColor Green
Write-Host "  Size:  ${exeSize} MB (exe) / ${totalSize} MB (total folder)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Folder contents:" -ForegroundColor Cyan
Write-Host "  ClearVid.exe     - Double-click to run" -ForegroundColor Gray
Write-Host "  _internal/       - Python runtime (PyInstaller)" -ForegroundColor Gray
Write-Host "  ffmpeg.exe       - FFmpeg binary" -ForegroundColor Gray
Write-Host "  vendor/          - Bundled basicsr" -ForegroundColor Gray
Write-Host "  outputs/         - Export output directory" -ForegroundColor Gray
Write-Host "  weights/         - Model weights (downloaded on demand)" -ForegroundColor Gray
Write-Host ""
Write-Host "First run: user will see a console wizard that downloads" -ForegroundColor Yellow
Write-Host "~2.8 GB of dependencies (PyTorch, PySide6, etc.) into lib/" -ForegroundColor Yellow
Write-Host ""
Write-Host "To distribute: ZIP the entire ClearVid/ folder." -ForegroundColor Cyan
