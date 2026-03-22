param(
    [ValidateSet("gui", "env", "sample-preview", "launcher")]
    [string]$Mode = "gui"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-PythonCommand {
    # 1. Portable embedded python
    $portablePython = Join-Path $repoRoot "python\python.exe"
    if (Test-Path $portablePython) {
        return @($portablePython)
    }
    # 2. venv
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return @($venvPython)
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3.13")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python is not available on PATH."
}

function Invoke-Python {
    param(
        [string[]]$Arguments
    )

    $pythonCommand = @(Get-PythonCommand)
    if ($pythonCommand.Length -eq 1) {
        & $pythonCommand[0] @Arguments
        return
    }

    & $pythonCommand[0] @($pythonCommand[1..($pythonCommand.Length - 1)]) @Arguments
}

try {
    Invoke-Python -Arguments @("-c", "import clearvid.app.gui") | Out-Null
} catch {
    # In development mode, install from source
    $libDir = Join-Path $repoRoot "lib"
    if (-not (Test-Path $libDir)) {
        Invoke-Python -Arguments @("-m", "pip", "install", "-e", ".[gui,media]")
    }
}

# Add lib/ to PYTHONPATH for portable mode
$libDir = Join-Path $repoRoot "lib"
if (Test-Path $libDir) {
    $env:PYTHONPATH = "$libDir;$env:PYTHONPATH"
}
$env:CLEARVID_ROOT = $repoRoot

switch ($Mode) {
    "env" {
        Invoke-Python -Arguments @("-m", "clearvid.app.cli", "env")
    }
    "sample-preview" {
        $samplePath = Join-Path $repoRoot "samples\sample.mp4"
        $outputPath = Join-Path $repoRoot "outputs\sample_fhd_preview20s.mp4"
        Invoke-Python -Arguments @(
            "-m", "clearvid.app.cli", "run",
            $samplePath,
            "--output", $outputPath,
            "--target-profile", "fhd",
            "--preview-seconds", "20"
        )
    }
    "launcher" {
        Invoke-Python -Arguments @("-m", "clearvid.app.bootstrap.launcher")
    }
    default {
        Invoke-Python -Arguments @("-m", "clearvid.app.cli", "gui")
    }
}
