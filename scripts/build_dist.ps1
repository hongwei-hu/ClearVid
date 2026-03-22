<#
.SYNOPSIS
    Build a portable ClearVid distribution package (ZIP).

.DESCRIPTION
    Creates a self-contained directory containing:
      - Python Embeddable (downloaded if not cached)
      - ClearVid source code
      - FFmpeg binaries (must be pre-placed or on PATH)
      - Launcher scripts
      - Empty lib/ placeholder (dependencies installed at first run)

    The result is a ZIP file that users can extract and run directly.

.PARAMETER OutputDir
    Directory where the distribution will be built. Default: dist/

.PARAMETER PythonVersion
    Python Embeddable version to bundle. Default: 3.13.4

.PARAMETER SkipPythonDownload
    If set, assumes python/ directory already exists in OutputDir.

.EXAMPLE
    .\build_dist.ps1
    .\build_dist.ps1 -OutputDir C:\builds\clearvid -PythonVersion 3.13.4
#>

param(
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "dist"),
    [string]$PythonVersion = "3.13.4",
    [switch]$SkipPythonDownload
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ClearVid Distribution Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$distName = "ClearVid"
$distDir = Join-Path $OutputDir $distName

# Clean previous build
if (Test-Path $distDir) {
    Write-Host "[1/6] Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $distDir
}
New-Item -ItemType Directory -Force -Path $distDir | Out-Null

# ---------------------------------------------------------------------------
# Step 1: Download Python Embeddable
# ---------------------------------------------------------------------------
$pythonDir = Join-Path $distDir "python"
if (-not $SkipPythonDownload) {
    Write-Host "[2/6] Downloading Python $PythonVersion Embeddable..." -ForegroundColor Yellow
    $pyZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    $pyZipPath = Join-Path $OutputDir "python-embed.zip"

    if (-not (Test-Path $pyZipPath)) {
        Invoke-WebRequest -Uri $pyZipUrl -OutFile $pyZipPath -UseBasicParsing
    }
    Expand-Archive -Path $pyZipPath -DestinationPath $pythonDir -Force

    # Enable pip: uncomment "import site" in python*._pth
    $pthFile = Get-ChildItem -Path $pythonDir -Filter "python*._pth" | Select-Object -First 1
    if ($pthFile) {
        $content = Get-Content $pthFile.FullName -Raw
        $content = $content -replace '#\s*import site', 'import site'
        # Add lib/ and source root to paths
        $content += "`n..\lib`n..`n"
        Set-Content -Path $pthFile.FullName -Value $content -NoNewline
    }

    # Install pip into embedded Python
    Write-Host "       Installing pip..." -ForegroundColor Gray
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $getPipPath = Join-Path $OutputDir "get-pip.py"
    if (-not (Test-Path $getPipPath)) {
        Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -UseBasicParsing
    }
    & (Join-Path $pythonDir "python.exe") $getPipPath --no-warn-script-location 2>&1 | Out-Null
} else {
    Write-Host "[2/6] Skipping Python download (--SkipPythonDownload)" -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# Step 2: Copy ClearVid source
# ---------------------------------------------------------------------------
Write-Host "[3/6] Copying ClearVid source..." -ForegroundColor Yellow

$sourceItems = @(
    "clearvid",
    "pyproject.toml",
    "README.md",
    "Start_ClearVid_GUI.bat",
    "scripts"
)
foreach ($item in $sourceItems) {
    $src = Join-Path $repoRoot $item
    $dst = Join-Path $distDir $item
    if (Test-Path $src -PathType Container) {
        # Copy directory, exclude __pycache__
        Copy-Item -Path $src -Destination $dst -Recurse -Force
        Get-ChildItem -Path $dst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
}

# ---------------------------------------------------------------------------
# Step 3: Copy vendor (basicsr)
# ---------------------------------------------------------------------------
$vendorSrc = Join-Path $repoRoot "vendor"
if (Test-Path $vendorSrc) {
    Write-Host "       Copying vendor libraries..." -ForegroundColor Gray
    $vendorDst = Join-Path $distDir "vendor"
    Copy-Item -Path $vendorSrc -Destination $vendorDst -Recurse -Force
    Get-ChildItem -Path $vendorDst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
}

# ---------------------------------------------------------------------------
# Step 4: FFmpeg binaries
# ---------------------------------------------------------------------------
Write-Host "[4/6] Checking FFmpeg..." -ForegroundColor Yellow

$ffmpegSrc = Join-Path $repoRoot "ffmpeg.exe"
$ffprobeSrc = Join-Path $repoRoot "ffprobe.exe"

if (Test-Path $ffmpegSrc) {
    Copy-Item -Path $ffmpegSrc -Destination (Join-Path $distDir "ffmpeg.exe")
    Write-Host "       Copied ffmpeg.exe from repo root" -ForegroundColor Gray
} else {
    $sysFFmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($sysFFmpeg) {
        Copy-Item -Path $sysFFmpeg.Source -Destination (Join-Path $distDir "ffmpeg.exe")
        Write-Host "       Copied ffmpeg.exe from PATH: $($sysFFmpeg.Source)" -ForegroundColor Gray
    } else {
        Write-Host "       WARNING: ffmpeg.exe not found! Users must provide their own." -ForegroundColor Red
    }
}

if (Test-Path $ffprobeSrc) {
    Copy-Item -Path $ffprobeSrc -Destination (Join-Path $distDir "ffprobe.exe")
    Write-Host "       Copied ffprobe.exe from repo root" -ForegroundColor Gray
} else {
    $sysFFprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($sysFFprobe) {
        Copy-Item -Path $sysFFprobe.Source -Destination (Join-Path $distDir "ffprobe.exe")
        Write-Host "       Copied ffprobe.exe from PATH: $($sysFFprobe.Source)" -ForegroundColor Gray
    } else {
        Write-Host "       WARNING: ffprobe.exe not found!" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# Step 5: Create placeholder directories and lib stamp trigger
# ---------------------------------------------------------------------------
Write-Host "[5/6] Creating directory structure..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path (Join-Path $distDir "outputs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "weights") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "samples") | Out-Null

# Copy sample videos if they exist
$samplesSrc = Join-Path $repoRoot "samples"
if (Test-Path $samplesSrc) {
    Copy-Item -Path "$samplesSrc\*" -Destination (Join-Path $distDir "samples") -Force -ErrorAction SilentlyContinue
}

# Copy weight README
$weightReadme = Join-Path $repoRoot "weights\README.zh-CN.md"
if (Test-Path $weightReadme) {
    Copy-Item -Path $weightReadme -Destination (Join-Path $distDir "weights\README.zh-CN.md") -Force
}

# ---------------------------------------------------------------------------
# Step 6: Create ZIP
# ---------------------------------------------------------------------------
Write-Host "[6/6] Creating ZIP archive..." -ForegroundColor Yellow

$zipPath = Join-Path $OutputDir "$distName.zip"
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path $distDir -DestinationPath $zipPath -CompressionLevel Optimal

$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  Output: $zipPath" -ForegroundColor Green
Write-Host "  Size:   ${zipSize} MB" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Distribution contents:" -ForegroundColor Cyan
Write-Host "  python/          - Python Embeddable $PythonVersion" -ForegroundColor Gray
Write-Host "  clearvid/        - ClearVid source code" -ForegroundColor Gray
Write-Host "  scripts/         - Launcher scripts" -ForegroundColor Gray
Write-Host "  ffmpeg.exe       - FFmpeg binary (if found)" -ForegroundColor Gray
Write-Host "  Start_ClearVid_GUI.bat - Double-click to run" -ForegroundColor Gray
Write-Host ""
Write-Host "First-run will install ~2.8 GB of dependencies (PyTorch, PySide6, etc.)" -ForegroundColor Yellow
