#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Bootstrap per “filmocredit-pipeline” su Windows.

.PARAMETER Root
  Percorso assoluto dove creare (o dove già esiste) il progetto.
  Se omesso, viene usata la directory corrente + $Name.

.PARAMETER Name
  Nome della cartella progetto se --Root non è fornito.
  Default: filmocredit-pipeline
#>

param(
    [string]$Root = "",
    [string]$Name = "filmocredit-pipeline"
)

#################################################################
# 0. Funzioni di utilità
#################################################################

function Write-Info    { param($m) Write-Host "[*] $m" -ForegroundColor Cyan }
function Write-ErrorX  { param($m) Write-Host "[!] $m" -ForegroundColor Red }
function Write-Success { param($m) Write-Host "[√] $m" -ForegroundColor Green }

function Assert-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] `
              [Security.Principal.WindowsIdentity]::GetCurrent()
              ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Info "Elevating to administrator..."
        Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`" $($MyInvocation.UnboundArguments)" -Verb RunAs
        exit
    }
}

#################################################################
# 1. Percorso radice
#################################################################
if ($Root -eq "") {
    $Root = Join-Path -Path (Get-Location) -ChildPath $Name
}

$resolved = Resolve-Path -LiteralPath $Root -ErrorAction SilentlyContinue
if ($resolved) {
    $Root = $resolved.Path
} else {
    $Root = (New-Item -Path $Root -ItemType Directory -Force).FullName
}
Write-Info "Project root: $Root"

#################################################################
# 2. Privilegi amministrativi
#################################################################
Assert-Admin

#################################################################
# 3. winget o choco
#################################################################
$PackageTool = if (Get-Command winget -ErrorAction SilentlyContinue) { "winget" }
               elseif (Get-Command choco  -ErrorAction SilentlyContinue) { "choco" }
               else {
                   Write-ErrorX "Né winget né Chocolatey trovati. Installa uno dei due e riprova."; exit 1
               }
Write-Info "Userò $PackageTool per installare i pacchetti di sistema"

function Install-Pkg {
    param($idWinget,$idChoco)
    if ($PackageTool -eq "winget") {
        winget install --id $idWinget  --accept-package-agreements --accept-source-agreements -e | Out-Null
    } else {
        choco install $idChoco -y | Out-Null
    }
}

#################################################################
# 4. Pacchetti di sistema
#################################################################
Install-Pkg "Python.Python.3"    "python"
Install-Pkg "Gyan.FFmpeg"        "ffmpeg"
Install-Pkg "UB-Mannheim.TesseractOCR" "tesseract"

#################################################################
# 5. Creazione cartelle progetto
#################################################################
$dirs = @(
    "data\raw","data\clips","data\frames","data\ocr",
    "db",
    "scripts",
    "app"
) | ForEach-Object { Join-Path $Root $_ }

foreach ($d in $dirs) { New-Item -Path $d -ItemType Directory -Force | Out-Null }
Write-Info "Cartelle pronte"

#################################################################
# 6. Virtual env + requirements
#################################################################
$envPath = Join-Path $Root "env"
$pyExe   = (Get-Command python).Source
Write-Info "Creazione venv..."
& $pyExe -m venv $envPath

$reqFile = Join-Path $Root "requirements.txt"
if (-not (Test-Path $reqFile)) {
@'
ffmpeg-python
opencv-python
pytesseract
rapidfuzz
textdistance
pandas
sqlalchemy
sqlalchemy-utils
streamlit
tqdm
duckdb
tabulate
'@ | Out-File $reqFile -Encoding UTF8
}

& "$envPath\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
& "$envPath\Scripts\pip.exe" install -r $reqFile | Out-Null
Write-Info "Venv creato e dipendenze installate"

#################################################################
# 7. Banner finale
#################################################################
Write-Host "`n=============================================" -ForegroundColor DarkGreen
Write-Host "   READY!  Progetto installato in" -ForegroundColor Green
Write-Host "   $Root" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor DarkGreen
Write-Host "Per iniziare:"
Write-Host "   cd `"$Root`""
Write-Host "   .\env\Scripts\Activate.ps1"
Write-Host "   01_segment_video.py --help"
Write-Host "Buon lavoro!"
