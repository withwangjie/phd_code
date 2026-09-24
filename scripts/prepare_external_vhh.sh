#!/usr/bin/env bash
#
# prepare_external_vhh.sh -- two-pass preparation of the external VHH set
# (docs/PROTOCOL_AMENDMENTS.md A8). Run from anywhere; steps in order:
#
#   ./scripts/prepare_external_vhh.sh audit
#   ./scripts/prepare_external_vhh.sh pass1 --sabdab-summary sabdab_nano_summary_all.tsv \
#         --released-after YYYY-MM-DD --download
#   ./scripts/prepare_external_vhh.sh foldseek [--foldseek /path/to/foldseek] [--threads N]
#   ./scripts/deploy_launch.sh --stop-after queue_freeze
#   ./scripts/prepare_external_vhh.sh pass2 --sabdab-summary sabdab_nano_summary_all.tsv \
#         --released-after YYYY-MM-DD --run-dir <that run directory>
#   ./scripts/deploy_launch.sh            # fresh formal run, pair table unchanged
#
# Honours QP_DATA_ROOT and QP_EXTERNAL_VHH_SOURCE_DIR. Installs nothing; reads
# no experimental outcome.
#
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -f "${REPO_ROOT}/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/Scripts/activate"
fi
exec python -m nanoqc.data.prepare_external_vhh "$@"
