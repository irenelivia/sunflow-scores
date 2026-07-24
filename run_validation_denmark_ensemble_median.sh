#!/usr/bin/env bash
# =============================================================================
# run_validation_denmark.sh — year-long validation over the Denmark domain
# =============================================================================
# Runs the full-year (2025) solar nowcast validation for a single model version,
# always restricted to the Denmark bounding box, writing one scores_YYYYMMDD.csv
# per day into a per-version output directory.
#
# Usage:
#     ./run_validation_denmark.sh v1.0.0
#     ./run_validation_denmark.sh v1.0.1
#
# The model version ($1) selects the nowcast input directory and the output
# directory; the observation directory and the Denmark bbox are the same for
# every version so the two runs are directly comparable.
# =============================================================================
set -euo pipefail

# HDF5 file locking must be disabled in the environment BEFORE Python/uv starts,
# since the HDF5 C library reads this variable only at initialization time.
# Without this, network mount reads (NFS/Lustre) spuriously fail with errno 121
# "unable to lock file" during data loading or scoring in some months.
export HDF5_USE_FILE_LOCKING=FALSE

# -----------------------------------------------------------------------------
# 0. Arguments and resolved paths
# -----------------------------------------------------------------------------
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <model-version>   e.g. $0 v1.0.0"
  exit 1
fi

# Denmark bounding box: LON_MIN LAT_MIN LON_MAX LAT_MAX
BBOX=(7.5 54.5 15.5 58.0)

nwc_base=/dmidata/projects/energivejr-data/nowcasts/$VERSION
obs_base=/dmidata/projects/energivejr-data/msg/2025/KNMI_MSGCPP_reproj_NW_EUROPE
tmp_out_base=/scratch/ikr/sunflow_validation_scores/2025_denmark/$VERSION
final_out_base=/dmidata/projects/energivejr-data/sunflow_validation_scores/$VERSION/2025_denmark/
mkdir -p "$tmp_out_base" "$final_out_base"

logfile="$tmp_out_base/run_validation_denmark.log"
# live console + log file
exec > >(stdbuf -oL tee -a "$logfile") 2>&1

echo "============================================================"
echo "  Denmark-domain validation"
echo "  Version  : $VERSION"
echo "  Nowcasts : $nwc_base"
echo "  Obs      : $obs_base"
echo "  Output   : $final_out_base"
echo "  BBox     : ${BBOX[*]}  (lon_min lat_min lon_max lat_max)"
echo "============================================================"


# -----------------------------------------------------------------------------
# 1. Loop months and days
# -----------------------------------------------------------------------------
for y in 2025; do
  for m in $(seq -w 1 12); do
    ym=${y}${m}
    nwc_dir="$nwc_base/$ym"
    obs_dir="$obs_base/$ym"
    out_dir="$tmp_out_base/$m"
    mkdir -p "$out_dir"

    if [[ ! -d "$nwc_dir" ]]; then
      echo "SKIP $ym: no nowcast directory $nwc_dir"
      continue
    fi
    if [[ ! -d "$obs_dir" ]]; then
      echo "SKIP $ym: no obs directory $obs_dir"
      continue
    fi

    last_day=$(date -d "$y-$m-01 +1 month -1 day" +%d)
    for d in $(seq -w 1 "$last_day"); do
      sdate="$y-$m-$d"

      if ls "$final_out_base/$m"/scores_${y}${m}${d}* 1>/dev/null 2>&1; then
        echo "SKIP $sdate: already done (found in final output dir)"
        continue
      fi

      any_nwc_file=$(ls "$nwc_dir"/SolarNowcast_${y}${m}${d}*.nc 2>/dev/null | head -n 1 || true)
      if [[ -z "$any_nwc_file" ]]; then
        echo "SKIP $sdate: no nowcast file in $nwc_dir"
        continue
      fi
      any_obs_file=$(ls "$obs_dir"/NetCDF4_sds_${y}-${m}-${d}T*.nc 2>/dev/null | head -n 1 || true)
      if [[ -z "$any_obs_file" ]]; then
        echo "SKIP $sdate: no obs file in $obs_dir"
        continue
      fi

      echo "=== RUN $sdate ==="
      echo "uv run python run_validation.py \\"
      echo "  --start $sdate --end $sdate \\"
      echo "  --nwc-dir $nwc_dir \\"
      echo "  --obs-dir $obs_dir \\"
      echo "  --output-dir $out_dir \\"
      echo "  --bbox ${BBOX[*]} \\"
      echo "  --nowcast_ghi_var GHI_probabilistic_advection_mean --obs_ghi_var sds --obs_cs_ghi_var sds_cs"

      uv run python run_validation.py \
        --start "$sdate" --end "$sdate" \
        --nwc-dir "$nwc_dir" \
        --obs-dir "$obs_dir" \
        --output-dir "$out_dir" \
        --bbox "${BBOX[@]}" \
        --nowcast_ghi_var GHI_probabilistic_advection_mean \
        --obs_ghi_var sds \
        --obs_cs_ghi_var sds_cs || {
          echo "ERROR $sdate: run_validation failed"
          continue
        }
      echo "=== DONE $sdate ==="

      # copy each day output to final NFS immediately to avoid data loss
      rsync -av --remove-source-files "$out_dir/" "$final_out_base/$m/"
    done
  done
done

# final sync
rsync -av --remove-source-files "$tmp_out_base/" "$final_out_base/"

kill "$KERB_PID" 2>/dev/null || true
echo "ALL DONE ($VERSION): $(date)"
