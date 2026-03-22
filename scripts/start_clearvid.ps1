param(
    [ValidateSet("gui", "env", "sample-preview")]
    [string]$Mode = "gui"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-PythonCommand {
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
    Invoke-Python -Arguments @("-m", "pip", "install", "-e", ".[gui,media]")
}

switch ($Mode) {
    "env" {
        Invoke-Python -Arguments @("-m", "clearvid.app.cli", "env")
    }
    "sample-preview" {
        $samplePath = Join-Path $repoRoot "samples\480P_2000K_306841291pigmask.mp4"
        $outputPath = Join-Path $repoRoot "outputs\480P_2000K_306841291pigmask_fhd_preview20s.mp4"
        Invoke-Python -Arguments @(
            "-m", "clearvid.app.cli", "run",
            $samplePath,
            "--output", $outputPath,
            "--target-profile", "fhd",
            "--preview-seconds", "20"
        )
    }
    default {
        Invoke-Python -Arguments @("-m", "clearvid.app.cli", "gui")
    }
}
