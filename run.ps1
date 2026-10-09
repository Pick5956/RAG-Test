<#
.SYNOPSIS  Start RAG Lab at http://127.0.0.1:8765 (open it in a browser; Ctrl+C stops it).
.PARAMETER Port  Use another port (default 8765).
#>
param([int]$Port = 8765)

Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "The .venv environment is missing. Run setup.ps1 first." }

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:APP_PORT = "$Port"
Write-Host "RAG Lab -> http://127.0.0.1:$Port   (Ctrl+C to stop)" -ForegroundColor Green
& $py app.py
