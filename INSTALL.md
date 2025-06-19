# FilmoCredit - Installation Guide

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

## 📁 Installation Location

FilmoCredit installs in the same directory where you run the installer:

```
📁 Your-Folder/                    
├── 📄 install.ps1 (or install.sh) # The installer script
├── 📄 run-filmocredit.bat/.sh     # Runner (created automatically)
└── 📁 FilmoCredit/                # Installation directory (created automatically)
    ├── 📁 .venv/                  # Virtual environment
    ├── 📄 app.py                  # FilmoCredit application
    └── 📄 FilmoCredit.bat         # Launcher
```

## 🎯 What the Installer Does

### Automatic Detection
- Platform: Windows, Linux, macOS
- Python Version: Requires Python 3.9+
- GPU Support: Automatically detects NVIDIA GPU and CUDA
- Dependencies: Installs PyTorch, PaddleOCR, and other requirements

### Installation Types
- **GPU Version**: If NVIDIA GPU + CUDA 12.6 detected
- **CPU Version**: If no GPU/CUDA detected
- **Self-Contained**: Creates isolated virtual environment

## ⚙️ Setup Requirements

### GPU Prerequisites (Optional)
For GPU acceleration, install **before** running the installer:
1. **NVIDIA GPU** with 4GB+ VRAM
2. **Latest NVIDIA drivers** 
3. **CUDA 12.6** from [NVIDIA website](https://developer.nvidia.com/cuda-12-6-0-download-archive)

### System Prerequisites
Most systems already have these, but install if missing:
1. **Python 3.11+** (preferably 3.11) from [python.org](https://python.org/downloads/)
   - On Windows: Check "Add Python to PATH" during installation
2. **Git** from [git-scm.com](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### Required Data Files

#### 1. Video Files
Place your video files in: **`FilmoCredit/data/raw/`**

You will already find a sample video named TEST.mp4 in the `data/raw` folder to let you test the software.

Supported formats: `.mp4`, `.mkv`, `.avi`, `.mov`

#### 2. IMDB Database (Required)
1. **Download**: `name.basics.tsv.gz` from https://datasets.imdbws.com/
2. **Extract the file**: Open and extract the content from the archive: you will get a file named `name.basics.tsv`
3. **Place**: The extracted `name.basics.tsv` file in **`FilmoCredit/db/`**

#### 3. Azure AI Configuration (Required)
The system currently works only with Azure AI models. Create a `.env` file in the **`FilmoCredit/`** root folder with your Azure credentials:

```
📁 FilmoCredit/
├── 📄 .env                       # ← Azure AI configuration file
├── 📁 data/raw/                  # ← Video files
└── 📁 db/                        # ← IMDB database
    └── name.basics.tsv
```

The `.env` file should contain:
```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

## 🚀 Running FilmoCredit

### Windows
- **Desktop Shortcut**: Double-click the FilmoCredit shortcut
- **Runner Script**: Double-click `run-filmocredit.bat`
- **Direct**: Run `FilmoCredit.bat` in the FilmoCredit folder

### Linux / macOS
- **Runner Script**: `./run-filmocredit.sh`
- **Direct**: `cd FilmoCredit && ./filmocredit`

The application will:
1. Activate the virtual environment
2. Start the Streamlit web interface
3. Open your browser to http://localhost:8501

## 🔄 Updates

Run the installer again to update to the latest version.

## 🛠️ Advanced Options

### Force Reinstall
```bash
# Linux/macOS
./install.sh --force

# Windows
./install.ps1 -Force
```

### CPU-Only Install
```bash
# Linux/macOS
GPU_AVAILABLE=false ./install.sh

# Windows
# Edit install.ps1 and set $hasGPU = $false
```

## 🐛 Troubleshooting

### Python Not Found
Most systems include Python, but if not detected:
- Install Python 3.11+ from [python.org](https://python.org/downloads/)
- On Windows: Check "Add Python to PATH" during installation

### Permission Errors (Linux/macOS)
```bash
chmod +x install.sh
# If needed: sudo apt install python3 python3-pip python3-venv
```

### GPU Not Detected
1. Install NVIDIA drivers
2. Install CUDA 12.6
3. Verify with `nvidia-smi`

## 📋 System Requirements

### Minimum
- **OS**: Windows 10, macOS 10.15, Linux (Ubuntu 18.04+)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.11+

### GPU Acceleration (Optional)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: Version 12.6
- **Drivers**: Latest NVIDIA drivers

## 🆘 Support

For issues:
1. Check the [Issues](https://github.com/guglielmopescatore/filmocredit-pipeline/issues) page
2. Run installer with verbose output: `bash -x install.sh`
3. Include system info and error messages when reporting bugs
