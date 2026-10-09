<#
.SYNOPSIS
  One-shot setup for RAG Lab (Windows PowerShell): venv, packages, embedding model. Documents are added from the web page (sample data: dataset/).

.PARAMETER GpuOcr    Create .venv-paddle with paddlepaddle-gpu (fast OCR on an NVIDIA GPU).
.PARAMETER CpuOcr    Install CPU PaddleOCR into .venv instead (slow OCR, no second venv).
.PARAMETER DryRun    Print every command without running it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup.ps1 -GpuOcr
#>
param([switch]$GpuOcr, [switch]$CpuOcr, [switch]$DryRun)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }

function Run([string]$Exe, [string[]]$CommandArgs) {
    Write-Host ("    " + $Exe + " " + ($CommandArgs -join " ")) -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit code $LASTEXITCODE): $Exe" }
}

# --- checks -----------------------------------------------------------------------------------------------
Step "Checking the environment"
if ($PSScriptRoot -match '[^\x00-\x7F]') {
    throw "This folder path contains non-ASCII characters (e.g. Thai). PaddleOCR cannot load its models from such a path. Move the project to an ASCII-only path, e.g. C:\work\rag-lab."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python not found. Install Python 3.11 and tick 'Add to PATH'." }
$pyVersion = (& python --version) 2>&1
Write-Host "    $pyVersion (developed and tested on 3.11)"
if ($pyVersion -notmatch "3\.(10|11|12)\.") { Write-Warning "Python 3.10-3.12 expected; other versions are untested." }
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) { Write-Host "    NVIDIA GPU driver found: installing the CUDA build of PyTorch" }
else { Write-Warning "nvidia-smi not found: installing the CPU build of PyTorch (embedding will be slow)." }

# --- main environment ---------------------------------------------------------------------------------------
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Step "Creating the main environment (.venv)"
if (-not (Test-Path $py)) { Run "python" @("-m", "venv", ".venv") } else { Write-Host "    .venv already exists" }
Run $py @("-m", "pip", "install", "--upgrade", "pip")

Step "Installing PyTorch"
if ($hasNvidia) { Run $py @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128") }
else { Run $py @("-m", "pip", "install", "torch") }

Step "Installing the other packages (requirements.txt)"
Run $py @("-m", "pip", "install", "-r", "requirements.txt")

# --- OCR (optional) -------------------------------------------------------------------------------------------
if ($GpuOcr) {
    Step "Creating the GPU OCR environment (.venv-paddle)"
    $pyGpu = Join-Path $PSScriptRoot ".venv-paddle\Scripts\python.exe"
    if (-not (Test-Path $pyGpu)) { Run "python" @("-m", "venv", ".venv-paddle") }
    Run $pyGpu @("-m", "pip", "install", "--upgrade", "pip")
    Run $pyGpu @("-m", "pip", "install", "paddlepaddle-gpu==3.2.1", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    Run $pyGpu @("-m", "pip", "install", "paddleocr==3.7.0")
}
elseif ($CpuOcr) {
    Step "Installing CPU OCR into .venv"
    Run $py @("-m", "pip", "install", "-r", "requirements-ocr-cpu.txt")
}
else { Write-Host "`n    (OCR not installed: scanned PDFs and images will fail until you re-run with -GpuOcr or -CpuOcr)" }

# --- embedding model ---------------------------------------------------------------------------------------------
Step "Downloading the embedding model (Qwen3-Embedding-0.6B, about 1.2 GB)"
if (-not (Test-Path "Qwen3-Embedding-0.6B\model.safetensors")) {
    Run $py @("-c", "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Embedding-0.6B', local_dir='Qwen3-Embedding-0.6B')")
} else { Write-Host "    already downloaded" }

# --- 3D globe library (one file, MIT licence) ---------------------------------------------------------------
Step "Downloading three.js for the 3D globe tab (about 0.7 MB)"
if (-not (Test-Path "static\three.module.min.js")) {
    Write-Host "    static\three.module.min.js  <-  cdn.jsdelivr.net/npm/three@0.160.0"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force static | Out-Null
        Invoke-WebRequest "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js" -OutFile "static\three.module.min.js"
    }
} else { Write-Host "    already downloaded" }

Write-Host "`nDone. Start the app with:  powershell -ExecutionPolicy Bypass -File run.ps1" -ForegroundColor Green
if ($DryRun) { Write-Host "(dry run: nothing was executed)" -ForegroundColor Yellow }
