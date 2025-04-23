#!/usr/bin/env bash
# setup_project.sh: bootstrap filmocredit pipeline project
set -euo pipefail

# Default values
NAME="filmocredit-pipeline"
ROOT=""

# Usage function
usage() {
  cat <<EOF
Usage: $0 [--root /absolute/path] [--name project-name]

  --root    Absolute or relative path where the project will be created
  --name    Folder name if --root is not given (default: filmocredit-pipeline)
EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Determine project directory
if [[ -n "$ROOT" ]]; then
  PROJECT_DIR="$ROOT"
else
  PROJECT_DIR="$PWD/$NAME"
fi

# Create project directory if missing
if [[ ! -d "$PROJECT_DIR" ]]; then
  mkdir -p "$PROJECT_DIR"
  echo "Created project directory: $PROJECT_DIR"
else
  echo "Project directory already exists: $PROJECT_DIR"
fi

# Create required folder tree
for dir in data/raw data/clips data/frames data/ocr db scripts app; do
  if [[ ! -d "$PROJECT_DIR/$dir" ]]; then
    mkdir -p "$PROJECT_DIR/$dir"
    echo "Created directory: $PROJECT_DIR/$dir"
  fi
done

# Install OS dependencies (idempotent)
echo "Installing OS dependencies..."
sudo apt-get update && sudo apt-get install -y \
  ffmpeg tesseract-ocr tesseract-ocr-ita \
  python3 python3-venv build-essential

echo "OS dependencies installed."

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
VENV_DIR="$PROJECT_DIR/env"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
  echo "Created virtual environment at $VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate and install Python requirements
REQ_FILE="$PROJECT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  cat <<EOF > "$REQ_FILE"
# Add your Python dependencies here, e.g.
# requests
EOF
  echo "Initialized requirements.txt"
fi

# Activate venv and install requirements
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
pip install -r "$REQ_FILE"
deactivate

echo -e "\n\033[1;32mSETUP COMPLETE at $PROJECT_DIR\033[0m"
echo "To start working:"
echo "  cd \"$PROJECT_DIR\" && source env/bin/activate"
