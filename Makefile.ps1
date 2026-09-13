# Lightweight PowerShell task runner (GNU Make isn't native on Windows).
# Usage: ./Makefile.ps1 <task>   e.g. ./Makefile.ps1 test

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Task,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$venvPython = ".venv/Scripts/python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = "python" }

switch ($Task) {
    "venv" {
        python -m venv .venv
        & $venvPython -m pip install --upgrade pip
    }
    "install" {
        & $venvPython -m pip install -r requirements-dev.txt -e .
    }
    "lint" {
        & $venvPython -m ruff check src tests
        & $venvPython -m black --check src tests
    }
    "format" {
        & $venvPython -m black src tests
        & $venvPython -m ruff check --fix src tests
    }
    "test" {
        & $venvPython -m pytest -q @Rest
    }
    "download-data" {
        & $venvPython -m pdm.data.download @Rest
    }
    "train" {
        & $venvPython -m pdm.training.train @Rest
    }
    "seed-reference" {
        & $venvPython scripts/seed_reference_data.py @Rest
    }
    "serve" {
        & $venvPython -m uvicorn pdm.serving.app:app --reload --port 8000
    }
    "drift-check" {
        & $venvPython -m pdm.drift.run_drift_check @Rest
    }
    "build-serve" {
        docker build -f docker/serve.Dockerfile -t pdm-serve:local .
    }
    "build-train" {
        docker build -f docker/train.Dockerfile -t pdm-train:local .
    }
    "build-drift" {
        docker build -f docker/drift.Dockerfile -t pdm-drift:local .
    }
    "kind-up" {
        ./scripts/setup_kind.ps1
    }
    "kind-down" {
        kind delete cluster --name pdm
    }
    "rollback" {
        ./scripts/rollback.ps1 @Rest
    }
    default {
        Write-Host "Unknown task '$Task'. Available: venv, install, lint, format, test, download-data, train, seed-reference, serve, drift-check, build-serve, build-train, build-drift, kind-up, kind-down, rollback"
        exit 1
    }
}
