#!/bin/bash
#
# FilmoCredit Universal Installer
# Supports: Linux, macOS, Windows (via Git Bash/WSL)
#
# Usage: curl -sSL https://github.com/guglielmopescatore/filmocredit-pipeline/raw/refactoring-monorepo/install.sh | bash
#        or: ./install.sh
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Emojis with fallbacks
if [[ "$TERM" == *"color"* ]] || [[ "$COLORTERM" != "" ]]; then
    EMOJI_ROCKET="🚀"
    EMOJI_CHECK="✅"
    EMOJI_ERROR="❌"
    EMOJI_WARNING="⚠️"
    EMOJI_INFO="💡"
    EMOJI_PACKAGE="📦"
    EMOJI_GPU="🎮"
    EMOJI_CPU="🖥️"
else
    EMOJI_ROCKET="[ROCKET]"
    EMOJI_CHECK="[OK]"
    EMOJI_ERROR="[ERROR]"
    EMOJI_WARNING="[WARN]"
    EMOJI_INFO="[INFO]"
    EMOJI_PACKAGE="[PKG]"
    EMOJI_GPU="[GPU]"
    EMOJI_CPU="[CPU]"
fi

log_info() {
    echo -e "${BLUE}${EMOJI_INFO} $1${NC}"
}

log_success() {
    echo -e "${GREEN}${EMOJI_CHECK} $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}${EMOJI_WARNING} $1${NC}"
}

log_error() {
    echo -e "${RED}${EMOJI_ERROR} $1${NC}"
}

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux";;
        Darwin*)    PLATFORM="macos";;
        CYGWIN*|MINGW*|MSYS*) PLATFORM="windows";;
        *)          PLATFORM="unknown";;
    esac
    
    case "$(uname -m)" in
        x86_64|amd64) ARCH="x64";;
        aarch64|arm64) ARCH="arm64";;
        *) ARCH="x64";;  # Default to x64
    esac
    
    log_info "Detected platform: $PLATFORM-$ARCH"
}

# Check if NVIDIA GPU is available
check_gpu() {
    # Check if nvidia-smi is available and working
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            log_success "NVIDIA GPU detected"
            
            # Check for CUDA installation
            local cuda_installed=false
            local cuda_version=""
            
            # Check CUDA environment variables
            for var in CUDA_PATH CUDA_HOME CUDA_ROOT; do
                if [ -n "${!var}" ] && [ -d "${!var}" ]; then
                    cuda_installed=true
                    # Try to get CUDA version from nvcc
                    if [ -x "${!var}/bin/nvcc" ]; then
                        cuda_version=$(${!var}/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
                    fi
                    break
                fi
            done
            
            # Alternative: check common CUDA paths
            if [ "$cuda_installed" = false ]; then
                for cuda_path in /usr/local/cuda* /opt/cuda* /usr/cuda*; do
                    if [ -d "$cuda_path" ]; then
                        cuda_installed=true
                        if [ -x "$cuda_path/bin/nvcc" ]; then
                            cuda_version=$($cuda_path/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
                        fi
                        break
                    fi
                done
            fi
            
            if [ "$cuda_installed" = true ]; then
                if [ -n "$cuda_version" ]; then
                    log_success "CUDA $cuda_version detected"
                    # Check if version is compatible (12.x)
                    if echo "$cuda_version" | grep -q "^12\."; then
                        GPU_AVAILABLE=true
                        return 0
                    else
                        log_warning "CUDA $cuda_version detected, but version 12.6 is recommended"
                        log_info "FilmoCredit is optimized for CUDA 12.6"
                        echo -n "Continue with GPU installation anyway? (y/N): "
                        read -r response
                        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
                            GPU_AVAILABLE=true
                            return 0
                        fi
                    fi
                else
                    log_success "CUDA installation detected"
                    GPU_AVAILABLE=true
                    return 0
                fi
            else
                log_warning "NVIDIA GPU found but CUDA not detected!"
                log_info "Please install CUDA 12.6:"
                case $PLATFORM in
                    linux)
                        log_info "Ubuntu/Debian: https://developer.nvidia.com/cuda-downloads?target_os=Linux"
                        log_info "Or: sudo apt install nvidia-cuda-toolkit"
                        ;;
                    *)
                        log_info "https://developer.nvidia.com/cuda-12-6-0-download-archive"
                        ;;
                esac
                echo ""
                echo -n "Install CPU version instead? (Y/n): "
                read -r response
                if [ "$response" = "n" ] || [ "$response" = "N" ]; then
                    log_error "CUDA installation required for GPU version. Exiting..."
                    exit 1
                fi
            fi
        fi
    fi
    
    log_warning "No NVIDIA GPU detected, will install CPU version"
    GPU_AVAILABLE=false
    return 1
    return 1
}

# Check Python installation
check_python() {
    local python_cmd=""
    
    # Try different Python commands
    for cmd in python3 python python3.11 python3.10 python3.9; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            local major=$(echo $version | cut -d. -f1)
            local minor=$(echo $version | cut -d. -f2)
            
            if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
                python_cmd="$cmd"
                PYTHON_VERSION="$version"
                break
            fi
        fi
    done
    
    if [ -z "$python_cmd" ]; then
        log_error "Python 3.9+ not found. Please install Python first:"
        case $PLATFORM in
            linux)
                echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
                echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
                ;;
            macos)
                echo "  brew install python3"
                echo "  or download from: https://www.python.org/downloads/"
                ;;
            windows)
                echo "  Download from: https://www.python.org/downloads/"
                ;;
        esac
        exit 1
    fi
    
    PYTHON_CMD="$python_cmd"
    log_success "Found Python $PYTHON_VERSION at: $(which $python_cmd)"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $PLATFORM in
        linux)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update -q
                sudo apt-get install -y \
                    libgl1-mesa-glx \
                    libglib2.0-0 \
                    libsm6 \
                    libxext6 \
                    libfontconfig1 \
                    libxrender1 \
                    libgomp1 \
                    wget \
                    curl
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y \
                    mesa-libGL \
                    glib2 \
                    libSM \
                    libXext \
                    fontconfig \
                    libXrender \
                    libgomp \
                    wget \
                    curl
            else
                log_warning "Unknown package manager. You may need to install dependencies manually."
            fi
            ;;
        macos)
            if ! command -v brew >/dev/null 2>&1; then
                log_warning "Homebrew not found. Installing..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            ;;
        windows)
            log_info "Windows detected. Make sure you have Microsoft Visual C++ Redistributable installed."
            ;;
    esac
}

# Create installation directory
create_install_dir() {
    # Install in current directory instead of system directories
    INSTALL_DIR="$(pwd)/FilmoCredit"
    
    log_info "Installing to: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
}

# Download FilmoCredit source
download_source() {
    log_info "Downloading FilmoCredit source..."
    
    if [ -d ".git" ]; then
        log_info "Updating existing installation..."
        git pull
    else
        # Clone the specific branch
        git clone --branch refactoring-monorepo https://github.com/guglielmopescatore/filmocredit-pipeline.git .
    fi
}

# Create virtual environment and install dependencies
setup_environment() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment
    $PYTHON_CMD -m venv .venv
    
    # Activate virtual environment
    case $PLATFORM in
        windows)
            source .venv/Scripts/activate
            PIP_CMD=".venv/Scripts/pip"
            PYTHON_VENV=".venv/Scripts/python"
            ;;
        *)
            source .venv/bin/activate
            PIP_CMD=".venv/bin/pip"
            PYTHON_VENV=".venv/bin/python"
            ;;
    esac
    
    # Upgrade pip
    $PIP_CMD install --upgrade pip
    
    # Install dependencies based on GPU availability
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "${EMOJI_GPU} Installing GPU dependencies from requirements-gpu.txt..."
        $PIP_CMD install -r requirements-gpu.txt
        VARIANT="gpu"
    else
        log_info "${EMOJI_CPU} Installing CPU dependencies from requirements-cpu.txt..."
        $PIP_CMD install -r requirements-cpu.txt
        VARIANT="cpu"
    fi
    
    log_success "Environment setup complete!"
}

# Create launcher scripts
create_launchers() {
    log_info "Creating launcher scripts..."
    
    case $PLATFORM in
        windows)
            # Create Windows batch file
            cat > "FilmoCredit.bat" << 'EOF'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python main.py
pause
EOF
            
            # Create PowerShell script
            cat > "FilmoCredit.ps1" << 'EOF'
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
python main.py
EOF
            
            log_success "Created Windows launchers: FilmoCredit.bat, FilmoCredit.ps1"
            ;;
            
        *)
            # Create shell script
            cat > "filmocredit" << 'EOF'
#!/bin/bash
echo "🎬 Starting FilmoCredit..."
cd "$(dirname "$0")"
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo "✅ Starting web interface..."
echo "📱 Opening browser..."
python -m streamlit run app.py
EOF
            chmod +x filmocredit
            
            # Create desktop entry for Linux
            if [ "$PLATFORM" = "linux" ]; then
                mkdir -p "$HOME/.local/share/applications"
                cat > "$HOME/.local/share/applications/filmocredit.desktop" << EOF
[Desktop Entry]
Name=FilmoCredit
Comment=Film Credit Analysis Tool
Exec=$INSTALL_DIR/filmocredit
Icon=$INSTALL_DIR/icon.png
Terminal=false
Type=Application
Categories=AudioVideo;Video;
EOF
            fi
            
            log_success "Created launcher: filmocredit"
            ;;
    esac
}

# Add to PATH (optional)
add_to_path() {
    # Skip adding to PATH since we're installing locally
    # Instead, create a convenient runner script in the same directory as the installer
    
    if [ "$PLATFORM" != "windows" ]; then
        # Create a runner script next to the installer
        local runner_script="$(dirname "$INSTALL_DIR")/run-filmocredit.sh"
        cat > "$runner_script" << EOF
#!/bin/bash
echo "🎬 Starting FilmoCredit..."
cd "\$(dirname "\$0")/FilmoCredit"
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo "✅ Starting web interface..."
echo "📱 Opening browser..."
python -m streamlit run app.py
EOF
        chmod +x "$runner_script"
        log_info "Created convenient runner: $runner_script"
    fi
}

# Main installation function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════╗"
    echo "║          FilmoCredit Installer       ║"
    echo "║        Universal Cross-Platform     ║"
    echo "╚══════════════════════════════════════╝"
    echo -e "${NC}"
    
    detect_platform
    check_python
    check_gpu
    install_system_deps
    create_install_dir
    download_source
    setup_environment
    create_launchers
    add_to_path
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════╗"
    echo "║     ${EMOJI_ROCKET} Installation Complete! ${EMOJI_ROCKET}       ║"
    echo "╚══════════════════════════════════════╝"
    echo -e "${NC}"
    
    log_success "FilmoCredit installed successfully!"
    log_info "Installation directory: $INSTALL_DIR"
    log_info "Variant installed: $VARIANT"
    
    case $PLATFORM in
        windows)
            log_info "To run: Double-click FilmoCredit.bat or run FilmoCredit.ps1"
            ;;
        *)
            log_info "To run: filmocredit (if added to PATH) or $INSTALL_DIR/filmocredit"
            ;;
    esac
    
    echo ""
    log_info "For updates, simply run this installer again!"
}

# Run main function
main "$@"
