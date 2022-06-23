# Build script for windows

param (
    [switch]$clone = $false,
    [switch]$installVulkan = $false,
    [switch]$develop = $false,
    [switch]$install = $false,
    [string]$libsDir = "."
)

$ErrorActionPreference = "Stop"

$RepoURL = 'https://github.com/taichi-dev/taichi'

WriteInfo("Setting up Python environment")
python -m venv venv
. venv\Scripts\activate.ps1
python -m pip install wheel
python -m pip install -r requirements_dev.txt
if (-not $?) { exit 1 }
