#!/usr/bin/env bash
# =============================================================================
# run_validation.sh — parameterized solar nowcast validation runner
# =============================================================================
# Runs validation for a date range over one or more nowcast directories,
# writing one scores_YYYYMMDD.csv per day to the output directory.
#
# Supports both external users (with local data directories) and DMI internal
# use (with paths configured via run_validation_dmi.sh wrapper).
#
# Configuration via environment variables:
#   NWC_DIR           — nowcast input directory (default: ./nowcasts)
#   OBS_DIR           — observation input directory (default: ./observations)
#   OUTPUT_DIR        — validation output directory (default: ./results)
#   START_DATE        — start date YYYY-MM-DD (required, can override with --start)
#   END_DATE          — end date YYYY-MM-DD (required, can override with --end)
#   BBOX              — bounding box (optional; if set, adds --bbox lon_min lat_min lon_max lat_max)
#   USE_TMUX          — run in detached tmux session if true (default: false)
#   PYTHON_CMD        — Python command (default: "uv run python")
#   HDF5_USE_FILE_LOCKING — set to FALSE for networked filesystems (default: empty, let HDF5 decide)
#
# Usage (external user):
#   export START_DATE=2025-01-01 END_DATE=2025-01-02
#   export NWC_DIR=./my_nowcasts OBS_DIR=./my_observations
#   bash run_validation.sh
#
# Usage (with custom bbox):
#   export START_DATE=2025-01-01 END_DATE=2025-01-02
#   export BBOX="7.5 54.5 15.5 58.0"  # Denmark
#   bash run_validation.sh
#
# Usage (DMI internal, with tmux):
#   bash run_validation_dmi.sh v1.0.0
# =============================================================================
set -euo pipefail

# Default configuration for external users
NWC_DIR="${NWC_DIR:-.}"
OBS_DIR="${OBS_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
PYTHON_CMD="${PYTHON_CMD:-uv run python}"
BBOX="${BBOX:-}"
USE_TMUX="${USE_TMUX:-false}"

# Enable or disable HDF5 file locking (useful for network filesystems)
if [[ -n "${HDF5_USE_FILE_LOCKING:-}" ]]; then
  export HDF5_USE_FILE_LOCKING
fi

# Parse command-line overrides (allow --start, --end, --output-dir at CLI)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      START_DATE="$2"
      shift 2
      ;;
    --end)
      END_DATE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --nwc-dir)
      NWC_DIR="$2"
      shift 2
      ;;
    --obs-dir)
      OBS_DIR="$2"
      shift 2
      ;;
    --bbox)
      BBOX="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Validate required inputs
if [[ -z "${START_DATE:-}" ]]; then
  echo "ERROR: START_DATE not set (use --start YYYY-MM-DD or export START_DATE=YYYY-MM-DD)" >&2
  exit 1
fi
if [[ -z "${END_DATE:-}" ]]; then
  echo "ERROR: END_DATE not set (use --end YYYY-MM-DD or export END_DATE=YYYY-MM-DD)" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Solar nowcast validation"
echo "  Start date  : $START_DATE"
echo "  End date    : $END_DATE"
echo "  Nowcasts    : $NWC_DIR"
echo "  Obs         : $OBS_DIR"
echo "  Output      : $OUTPUT_DIR"
if [[ -n "$BBOX" ]]; then
  echo "  BBox        : $BBOX  (lon_min lat_min lon_max lat_max)"
fi
echo "============================================================"

# Build the run_validation.py command
RUN_CMD="$PYTHON_CMD run_validation.py"
RUN_CMD="$RUN_CMD --start $START_DATE --end $END_DATE"
RUN_CMD="$RUN_CMD --nwc-dir $NWC_DIR"
RUN_CMD="$RUN_CMD --obs-dir $OBS_DIR"
RUN_CMD="$RUN_CMD --output-dir $OUTPUT_DIR"
if [[ -n "$BBOX" ]]; then
  RUN_CMD="$RUN_CMD --bbox $BBOX"
fi

# Run in tmux if requested
if [[ "$USE_TMUX" == "true" ]]; then
  session_name="sunflow_validation_$(date +%s)"
  tmux new-session -d -s "$session_name" -x 200 -y 50 "$RUN_CMD"
  echo "Validation running in tmux session: $session_name"
  echo "Attach with: tmux attach-session -t $session_name"
else
  # Run directly in foreground
  eval "$RUN_CMD"
fi

echo "DONE: $(date)"
