#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-$ROOT_DIR/From_Finetuning_to_Attention_Inside_LLMs.ipynb}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
KERNEL_NAME="${KERNEL_NAME:-from-finetuning-to-attention}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (.venv - From Finetuning to Attention)}"
NOTEBOOK_TIMEOUT="${NOTEBOOK_TIMEOUT:-43200}"

install_python_if_missing() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    echo "Python found: $($PYTHON_BIN --version) at $PYTHON_BIN"
    return
  fi

  echo "python3 was not found. Attempting to install Python."

  if command -v brew >/dev/null 2>&1; then
    brew install python
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y python3 python3-pip
  else
    echo "Could not install Python automatically. Install Python 3 manually and rerun this script." >&2
    exit 1
  fi

  PYTHON_BIN="$(command -v python3)"
  echo "Python installed: $($PYTHON_BIN --version) at $PYTHON_BIN"
}

load_env_file() {
  if [ -f "$ROOT_DIR/.env" ]; then
    echo "Loading environment variables from .env"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  else
    echo ".env file not found. Continuing without local environment variables."
  fi
}

create_virtualenv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "Virtual environment already exists at $VENV_DIR"
  fi

  VENV_PYTHON="$VENV_DIR/bin/python"
  echo "Virtualenv Python: $($VENV_PYTHON --version)"
}

install_dependencies() {
  echo "Installing Jupyter and notebook dependencies into the virtual environment"
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install \
    notebook \
    nbconvert \
    ipykernel \
    torch \
    transformers \
    peft \
    pandas \
    matplotlib \
    seaborn \
    accelerate \
    bitsandbytes \
    datasets
}

install_jupyter_kernel() {
  echo "Installing virtualenv as a Jupyter kernel"
  "$VENV_PYTHON" -m ipykernel install \
    --user \
    --name "$KERNEL_NAME" \
    --display-name "$KERNEL_DISPLAY_NAME"
}

run_notebook() {
  if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Notebook not found: $NOTEBOOK_PATH" >&2
    exit 1
  fi

  echo "Running notebook end to end:"
  echo "$NOTEBOOK_PATH"
  "$VENV_DIR/bin/jupyter" nbconvert \
    --to notebook \
    --execute "$NOTEBOOK_PATH" \
    --inplace \
    --ExecutePreprocessor.timeout="$NOTEBOOK_TIMEOUT" \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME"
}

install_python_if_missing
load_env_file
create_virtualenv
install_dependencies
install_jupyter_kernel
run_notebook

echo "Notebook execution finished."
