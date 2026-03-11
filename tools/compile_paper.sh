#!/usr/bin/env bash
# Compile the METIS academic paper to PDF.
# Usage: bash tools/compile_paper.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")/paper"

cd "$PAPER_DIR"

echo "=== Compiling METIS paper ==="
echo "  Working directory: $(pwd)"
echo "  Figures:"
ls -la figures/*.pdf 2>/dev/null || echo "  (no figures found)"

# Two passes for cross-references
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

echo ""
echo "=== Compilation complete ==="
ls -la main.pdf
echo "Output: $PAPER_DIR/main.pdf"
