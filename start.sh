#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# PlateGenie AI — Startup Script
# ─────────────────────────────────────────────────────────────────────────────
# First time? Make this script executable:
#   chmod +x start.sh
#
# Then run it:
#   ./start.sh            # normal mode (Gradio)
#   ./start.sh --fallback # force fallback HTML interface
# ─────────────────────────────────────────────────────────────────────────────

# ── Color tokens ──────────────────────────────────────────────────────────────
RESET="\033[0m"
GREEN="\033[38;2;44;122;74m"       # #2C7A4A — primary forest green
GREEN_BOLD="\033[1;38;2;44;122;74m"
ACCENT="\033[38;2;76;175;80m"      # #4CAF50 — material green accent
DIM="\033[38;2;76;175;80m"
YELLOW="\033[33m"
RED="\033[38;2;231;76;60m"         # #e74c3c — danger red
RED_BOLD="\033[1;38;2;231;76;60m"
WHITE="\033[38;2;232;240;236m"
BOLD="\033[1m"
FAINT="\033[2m"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "  ${ACCENT}▸${RESET} $1"; }
success() { echo -e "  ${GREEN}✔${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail()    { echo -e "  ${RED}✘${RESET} $1"; }

# ── Banner ────────────────────────────────────────────────────────────────────
clear
echo ""
echo -e "  ${GREEN_BOLD}╔═══════════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${GREEN_BOLD}║                                                           ║${RESET}"
echo -e "  ${GREEN_BOLD}║${RESET}   ${BOLD}${WHITE}🍔  PlateGenie AI${RESET}                                   ${GREEN_BOLD}║${RESET}"
echo -e "  ${GREEN_BOLD}║${RESET}   ${FAINT}${ACCENT}Snap a photo. Know your nutrition.${RESET}               ${GREEN_BOLD}║${RESET}"
echo -e "  ${GREEN_BOLD}║                                                           ║${RESET}"
echo -e "  ${GREEN_BOLD}╚═══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Resolve script directory ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1 — Check Python 3 ──────────────────────────────────────────────────
info "Checking for Python 3…"

PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    # On some systems `python` is Python 3
    if python --version 2>&1 | grep -q "Python 3"; then
        PYTHON_BIN="python"
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    fail "Python 3 is not installed (or not on your PATH)."
    echo ""
    echo -e "  ${YELLOW}Install it from:${RESET}  https://www.python.org/downloads/"
    echo -e "  ${YELLOW}macOS (Homebrew):${RESET} brew install python3"
    echo -e "  ${YELLOW}Ubuntu/Debian:${RESET}    sudo apt install python3 python3-venv"
    echo ""
    exit 1
fi

PY_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
success "Found Python ${BOLD}${PY_VERSION}${RESET} (${PYTHON_BIN})"

# ── Step 2 — Virtual environment ──────────────────────────────────────────────
if [[ ! -d "venv" ]]; then
    info "No virtual environment found — creating one…"
    $PYTHON_BIN -m venv venv
    if [[ $? -ne 0 ]]; then
        fail "Could not create virtual environment."
        echo -e "  ${YELLOW}Try installing venv:${RESET}  sudo apt install python3-venv  (Linux)"
        exit 1
    fi
    success "Virtual environment created at ${BOLD}venv/${RESET}"
else
    success "Virtual environment already exists at ${BOLD}venv/${RESET}"
fi

# ── Step 3 — Activate virtual environment ────────────────────────────────────
info "Activating virtual environment…"
source venv/bin/activate
if [[ $? -ne 0 ]]; then
    fail "Failed to activate virtual environment."
    echo -e "  ${YELLOW}Try running manually:${RESET}  source venv/bin/activate"
    exit 1
fi
success "Virtual environment activated"

# ── Step 4 — Upgrade pip ─────────────────────────────────────────────────────
info "Upgrading pip…"
pip install --upgrade pip --quiet 2>/dev/null
if [[ $? -ne 0 ]]; then
    warn "pip upgrade failed (non-fatal, continuing…)"
else
    success "pip upgraded to $(pip --version | awk '{print $2}')"
fi

# ── Step 5 — Install requirements ─────────────────────────────────────────────
echo ""
info "Installing dependencies from requirements.txt…"
echo -e "  ${FAINT}─────────────────────────────────────────────────────────────${RESET}"

pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
    echo ""
    fail "Dependency installation failed."
    echo ""
    echo -e "  ${YELLOW}Common fixes:${RESET}"
    echo -e "    • Update pip:        ${FAINT}pip install --upgrade pip${RESET}"
    echo -e "    • Install build tools: ${FAINT}sudo apt install python3-dev build-essential${RESET}  (Linux)"
    echo -e "    • For torch issues:  ${FAINT}pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu${RESET}"
    echo -e "    • Check Python version: ${FAINT}python3 --version${RESET}  (need 3.8+)"
    echo ""
    exit 1
fi

echo -e "  ${FAINT}─────────────────────────────────────────────────────────────${RESET}"
success "All dependencies installed successfully"

# Show what was installed (top-level packages from requirements.txt)
echo ""
info "Installed packages:"
while IFS= read -r line; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Extract package name (before any version specifier)
    pkg=$(echo "$line" | sed 's/[>=<!].*//' | sed 's/\[.*\]//' | xargs)
    installed_ver=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    if [[ -n "$installed_ver" ]]; then
        echo -e "    ${ACCENT}●${RESET} ${pkg} ${FAINT}${installed_ver}${RESET}"
    else
        echo -e "    ${ACCENT}●${RESET} ${pkg} ${FAINT}(installed)${RESET}"
    fi
done < requirements.txt

# ── Optional: Model Configuration ────────────────────────────────────────────
# Uncomment to use a different YOLO model (default: yolov8s.pt)
# export PLATEGENIE_YOLO_MODEL="yolov8m.pt"
# Uncomment to adjust detection confidence (default: 0.20)
# export PLATEGENIE_YOLO_CONF="0.25"

# ── Step 6 — Launch the app ──────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN_BOLD}───────────────────────────────────────────────────────────────${RESET}"
echo -e "  ${GREEN_BOLD}  🌐  Launching PlateGenie AI…${RESET}"
echo -e "  ${GREEN_BOLD}───────────────────────────────────────────────────────────────${RESET}"
echo ""

# Pass through all CLI arguments (e.g. --fallback)
python app.py "$@"
