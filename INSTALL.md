# FilmoCredit - Universal Installation Guide

FilmoCredit now supports **cross-platform installation** with smart installers that automatically detect your system and install the appropriate dependencies.

## 🚀 Quick Install

### Linux / macOS
```bash
curl -sSL https://raw.githubusercontent.com/guglielmopescatore/filmocredit-pipeline/refactoring-monorepo/install.sh | bash
```

### Windows (PowerShell)
```powershell
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/guglielmopescatore/filmocredit-pipeline/refactoring-monorepo/install.ps1'))
```

### Manual Download
1. Download `install.sh` (Linux/macOS) or `install.ps1` (Windows)
2. Run the script:
   - Linux/macOS: `chmod +x install.sh && ./install.sh`
   - Windows: `powershell -ExecutionPolicy Bypass -File install.ps1`

## 🎯 What the Installer Does

### Automatic Detection
- ✅ **Platform**: Windows, Linux, macOS
- ✅ **Architecture**: x64, ARM64
- ✅ **Python Version**: Requires Python 3.9+
- ✅ **GPU Support**: Automatically detects NVIDIA GPU
- ✅ **Dependencies**: Installs system libraries as needed

### Smart Installation
- 🎮 **GPU Version**: If NVIDIA GPU + CUDA 12.6 detected → CUDA-enabled PyTorch + PaddlePaddle
- 🖥️ **CPU Version**: No GPU/CUDA detected → CPU-only PyTorch + PaddlePaddle
- 📦 **Self-Contained**: Creates isolated virtual environment
- 🔧 **Launchers**: Creates platform-appropriate shortcuts

### GPU Prerequisites ⚠️
**BEFORE running the installer for GPU support:**
1. **NVIDIA GPU** with 4GB+ VRAM
2. **Latest NVIDIA drivers** 
3. **CUDA 12.6** installed from [NVIDIA website](https://developer.nvidia.com/cuda-12-6-0-download-archive)

The installer will automatically detect and verify these requirements.

## 📁 Installation Locations

**FilmoCredit installs in the same directory where you run the installer:**

```
📁 Your-Folder/                    # Where you put install.ps1 or install.sh
├── 📄 install.ps1 (or install.sh) # The installer script
├── 📄 run-filmocredit.bat/.sh     # Convenient runner (created automatically)
└── 📁 FilmoCredit/                # Installation directory (created automatically)
    ├── 📁 .venv/                  # Virtual environment
    ├── 📄 main.py                 # FilmoCredit application
    ├── 📄 FilmoCredit.bat         # Windows launcher
    └── 📄 filmocredit             # Linux/macOS launcher
```

This **portable approach** means you can:
- ✅ Install anywhere you want
- ✅ Move the entire folder to another computer
- ✅ Have multiple installations
- ✅ Easy cleanup (just delete the folder)

## 🚀 Running FilmoCredit

### Windows
- **⭐ Desktop Shortcut**: Double-click the FilmoCredit shortcut (created automatically)
- **🎯 Runner Script**: Double-click `run-filmocredit.bat` next to the installer
- **📁 Direct**: Go to `FilmoCredit/` folder and run `FilmoCredit.bat`

### Linux / macOS
- **🎯 Runner Script**: `./run-filmocredit.sh` next to the installer
- **📁 Direct**: `cd FilmoCredit && ./filmocredit`

### What Happens When You Run FilmoCredit

All the launchers do the same thing automatically:
1. **Activate the virtual environment** (with all the Python dependencies)
2. **Start the Streamlit web interface** 
3. **Open your browser** to the FilmoCredit interface
4. **You're ready to analyze your videos!** 🎬

**Example of what you'll see:**
```
🎬 Starting FilmoCredit...
✅ Virtual environment activated
✅ Starting web interface...
📱 Opening browser...

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501

  Welcome to FilmoCredit! 🎬
  Upload your video file to start analyzing credits...
```

**The web interface will automatically open in your default browser!**

### First Time Setup
After installation, the first run might take a few extra minutes to:
- Download AI models (PaddleOCR, etc.)
- Set up the database
- Initialize the analysis pipeline

**Subsequent runs will be much faster!** ⚡

## 🔄 Updates

Simply run the installer again! It will:
- Detect existing installation
- Update to latest version
- Preserve your settings and data

## 🛠️ Advanced Options

### Custom Installation Path
```bash
# The installer always creates a "FilmoCredit" folder where you run it
# To install elsewhere, just move the installer file:

# Example: Install on Desktop
cp install.sh ~/Desktop/ && cd ~/Desktop/ && ./install.sh

# Example: Install on USB drive  
cp install.ps1 E:\ && cd E:\ && ./install.ps1
```

### Force Reinstall
```bash
# Linux/macOS
./install.sh --force

# Windows
./install.ps1 -Force
```

### CPU-Only Install (Force)
```bash
# Linux/macOS
GPU_AVAILABLE=false ./install.sh

# Windows
# Edit install.ps1 and set $hasGPU = $false
```

## 🐛 Troubleshooting

### Python Not Found
Install Python 3.9+ from [python.org](https://python.org/downloads/)

**Windows**: Make sure to check "Add Python to PATH"

### Permission Errors (Linux/macOS)
```bash
# Make installer executable
chmod +x install.sh

# Install system dependencies manually if needed
sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
brew install python3                                # macOS
```

### GPU Not Detected
1. Install NVIDIA drivers
2. Install CUDA 12.6
3. Run `nvidia-smi` to verify

### Firewall/Network Issues
If automatic download fails:
1. Download repository manually
2. Extract to installation directory
3. Run installer from that directory

## 📋 System Requirements

### Minimum
- **OS**: Windows 10, macOS 10.15, Linux (Ubuntu 18.04+)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.9 or higher

### GPU Acceleration (Optional)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: Version 12.6
- **Drivers**: Latest NVIDIA drivers

## 🆘 Support

If you encounter issues:
1. Check the [Issues](https://github.com/guglielmopescatore/filmocredit-pipeline/issues) page
2. Run installer with verbose output: `bash -x install.sh`
3. Include system info and error messages when reporting bugs
