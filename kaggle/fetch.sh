#!/usr/bin/env bash
# Download a finished Kaggle job's results into the repo.
#
# Usage:  bash kaggle/fetch.sh <job-slug>      (the slug is printed by kaggle/launch.sh)
#
# Experiment folders go to experiments/<Member>/expNN/ and the job's full log
# goes to kaggle/logs/<job-slug>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG="$1"
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME to your Kaggle username}"
DOWNLOAD_DIR="$(mktemp -d)"

uvx --with six kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$DOWNLOAD_DIR" --force

mkdir -p kaggle/logs
mv "$DOWNLOAD_DIR/$SLUG.log" kaggle/logs/
if [ -d "$DOWNLOAD_DIR/experiments" ]; then
    cp -r "$DOWNLOAD_DIR/experiments/." experiments/
    echo "Copied into experiments/:"
    find "$DOWNLOAD_DIR/experiments" -mindepth 2 -maxdepth 2 -type d | sed "s|$DOWNLOAD_DIR/||"
else
    echo "No experiment results in this job (setup check, or the job failed). See kaggle/logs/$SLUG.log"
fi
