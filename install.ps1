# FilmoCredit Universal Installer for Windows
# Usage: iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/guglielmopescatore/filmocredit-pipeline/refactoring-monorepo/install.ps1'))

param(
    [switch]$Force,
    [string]$InstallPath = $PWD.Path  # Install in current directory instead of system location
)

$ErrorActionPreference = "Stop"

# Colors and emojis
function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success($Message) {
    Write-ColorOutput "✅ $Message" "Green"
}

function Write-Info($Message) {
    Write-ColorOutput "💡 $Message" "Blue"
}

function Write-Warning($Message) {
    Write-ColorOutput "⚠️  $Message" "Yellow"
}

function Write-Error($Message) {
    Write-ColorOutput "❌ $Message" "Red"
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check Python installation
function Test-Python {
    $pythonCommands = @("python", "python3", "py")
    
    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>$null
            if ($version -match "Python (\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                
                if ($major -eq 3 -and $minor -ge 9) {
                    Write-Success "Found $version"
                    return $cmd
                }
            }
        } catch {
            continue
        }
    }
    
    Write-Error "Python 3.9+ not found!"
    Write-Info "Please install Python from: https://www.python.org/downloads/"
    Write-Info "Make sure to check 'Add Python to PATH' during installation"
    exit 1
}

# Check for NVIDIA GPU and CUDA
function Test-GPU {    # First check if NVIDIA GPU is present
    try {
        $null = & nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "NVIDIA GPU detected"
            
            # Check for CUDA installation
            $cudaInstalled = $false
            $cudaVersion = ""
            
            # Check CUDA environment variables
            $cudaPaths = @($env:CUDA_PATH, $env:CUDA_HOME, $env:CUDA_ROOT)
            foreach ($path in $cudaPaths) {
                if ($path -and (Test-Path $path)) {
                    $cudaInstalled = $true
                    # Try to get CUDA version
                    $nvccPath = Join-Path $path "bin\nvcc.exe"
                    if (Test-Path $nvccPath) {
                        try {
                            $nvccOutput = & $nvccPath --version 2>$null
                            if ($nvccOutput -match "release (\d+\.\d+)") {
                                $cudaVersion = $matches[1]
                            }
                        } catch { }
                    }
                    break
                }
            }
            
            # Alternative: check common CUDA installation paths
            if (-not $cudaInstalled) {
                $commonPaths = @(
                    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
                    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
                    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
                )
                foreach ($path in $commonPaths) {
                    if (Test-Path $path) {
                        $cudaInstalled = $true
                        if ($path -match "v(\d+\.\d+)") {
                            $cudaVersion = $matches[1]
                        }
                        break
                    }
                }
            }
            
            if ($cudaInstalled) {
                if ($cudaVersion) {
                    Write-Success "CUDA $cudaVersion detected"
                    # Check if version is compatible (12.x)
                    if ($cudaVersion -match "^12\.") {
                        return $true
                    } else {
                        Write-Warning "CUDA $cudaVersion detected, but version 12.6 is recommended"
                        Write-Info "FilmoCredit is optimized for CUDA 12.6"
                        $response = Read-Host "Continue with GPU installation anyway? (y/N)"
                        return ($response -eq "y" -or $response -eq "Y")
                    }
                } else {
                    Write-Success "CUDA installation detected"
                    return $true
                }
            } else {
                Write-Warning "NVIDIA GPU found but CUDA not detected!"
                Write-Info "Please install CUDA 12.6 from:"
                Write-Info "https://developer.nvidia.com/cuda-12-6-0-download-archive"
                Write-Info ""
                Write-Info "Or continue with CPU-only installation"
                $response = Read-Host "Install CPU version instead? (Y/n)"
                return ($response -eq "n" -or $response -eq "N")
            }
        }
    } catch {
        Write-Warning "No NVIDIA GPU detected, will install CPU version"
        return $false
    }
    
    return $false
}

# Download and extract from GitHub
function Get-FilmoCredit {
    param($InstallPath)
      Write-Info "Downloading FilmoCredit to current directory..."
    
    # Create FilmoCredit subdirectory in current location
    $InstallPath = Join-Path $InstallPath "FilmoCredit"
    
    if (Test-Path $InstallPath) {
        if ($Force) {
            Remove-Item $InstallPath -Recurse -Force
        } else {
            Write-Warning "Installation directory already exists: $InstallPath"
            $response = Read-Host "Overwrite? (y/N)"
            if ($response -ne "y" -and $response -ne "Y") {
                exit 1
            }
            Remove-Item $InstallPath -Recurse -Force
        }
    }
    
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
    Set-Location $InstallPath
      # Download latest release or clone repo
    try {
        # Try to download latest release first
        $apiUrl = "https://api.github.com/repos/guglielmopescatore/filmocredit-pipeline/releases/latest"
        $release = Invoke-RestMethod -Uri $apiUrl
        $zipUrl = $release.zipball_url
        
        $zipPath = "$env:TEMP\filmocredit.zip"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
        
        Expand-Archive -Path $zipPath -DestinationPath $InstallPath
        Remove-Item $zipPath
        
        # Move files from extracted folder to install directory
        $extractedFolder = Get-ChildItem $InstallPath -Directory | Select-Object -First 1
        Move-Item "$($extractedFolder.FullName)\*" $InstallPath
        Remove-Item $extractedFolder.FullName
        
    } catch {
        Write-Warning "Could not download release, trying git clone..."
        try {
            & git clone --branch refactoring-monorepo https://github.com/guglielmopescatore/filmocredit-pipeline.git $InstallPath
        } catch {
            Write-Error "Failed to download FilmoCredit. Please check your internet connection."
            exit 1
        }
    }
}

# Setup Python environment
function Set-Environment {
    param($PythonCmd, $HasGPU)
    
    Write-Info "Setting up Python environment..."
      # Create virtual environment
    & $PythonCmd -m venv .venv
    
    $pipCmd = ".\.venv\Scripts\pip.exe"
      # Upgrade pip
    & $pipCmd install --upgrade pip
    
    if ($HasGPU) {
        Write-Info "🎮 Installing GPU dependencies from requirements-gpu.txt..."
        & $pipCmd install -r requirements-gpu.txt
        $variant = "GPU"
    } else {
        Write-Info "🖥️ Installing CPU dependencies from requirements-cpu.txt..."
        & $pipCmd install -r requirements-cpu.txt
        $variant = "CPU"
    }
    
    Write-Success "Environment setup complete! ($variant variant)"
}

# Create launchers
function New-Launchers {
    Write-Info "Creating launcher scripts..."
      # Batch file launcher
    @"
@echo off
echo 🎬 Starting FilmoCredit...
cd /d "%~dp0"
call .venv\Scripts\activate
echo ✅ Virtual environment activated
echo ✅ Starting web interface...
python -m streamlit run app.py --server.headless true --server.port 8501
pause
"@ | Out-File -FilePath "FilmoCredit.bat" -Encoding ASCII
      # PowerShell launcher
    @"
param([string[]]`$Arguments)
Write-Host "🎬 Starting FilmoCredit..." -ForegroundColor Blue
Set-Location `$PSScriptRoot
& .\.venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host "✅ Starting web interface..." -ForegroundColor Green
python -m streamlit run app.py --server.headless true --server.port 8501
"@ | Out-File -FilePath "FilmoCredit.ps1" -Encoding UTF8
      # Desktop shortcut in the same directory as the installer
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut("$desktopPath\FilmoCredit.lnk")
    $shortcut.TargetPath = "$InstallPath\FilmoCredit.bat"
    $shortcut.WorkingDirectory = $InstallPath
    $shortcut.IconLocation = "$InstallPath\icon.ico"
    $shortcut.Save()
      # Also create a runner in the same directory as the installer
    $runnerPath = Join-Path (Split-Path $InstallPath -Parent) "run-filmocredit.bat"
    @"
@echo off
echo 🎬 Starting FilmoCredit...
cd /d "%~dp0\FilmoCredit"
call .venv\Scripts\activate
echo ✅ Virtual environment activated  
echo ✅ Starting web interface...
echo 📱 Opening browser...
python -m streamlit run app.py
"@ | Out-File -FilePath $runnerPath -Encoding ASCII
    
    Write-Success "Created launchers, desktop shortcut, and runner script"
}

# Main installation
function Install-FilmoCredit {
    Write-Host @"
╔══════════════════════════════════════╗
║          FilmoCredit Installer       ║
║            Windows Version           ║
╚══════════════════════════════════════╝
"@ -ForegroundColor Blue

    $pythonCmd = Test-Python
    $hasGPU = Test-GPU
    
    Write-Info "Installing to: $InstallPath"
    
    Get-FilmoCredit -InstallPath $InstallPath
    Set-Environment -PythonCmd $pythonCmd -HasGPU $hasGPU
    New-Launchers
    
    Write-Host @"
╔══════════════════════════════════════╗
║     🚀 Installation Complete! 🚀       ║
╚══════════════════════════════════════╝
"@ -ForegroundColor Green

    Write-Success "FilmoCredit installed successfully!"
    Write-Info "Installation directory: $InstallPath"
    Write-Info "To run: Double-click the desktop shortcut or FilmoCredit.bat"
    Write-Info "For updates, simply run this installer again!"
}

# Run installation
Install-FilmoCredit
