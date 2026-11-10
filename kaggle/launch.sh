#!/usr/bin/env bash
# Send a training job to a Kaggle T4 GPU from the terminal (no notebook upload needed).
#
# Usage:  bash kaggle/launch.sh <Member> <experiments> [git-ref]
#   bash kaggle/launch.sh Hassan check main            # only test setup + GPU (a few minutes)
#   bash kaggle/launch.sh Hassan 1-9 v1-experiments    # run exp01-exp09 one after another
#   bash kaggle/launch.sh Hassan 10 v1-experiments     # run exp10
#
# Needs: KAGGLE_USERNAME set and a Kaggle API token in ~/.kaggle/access_token.
set -euo pipefail
cd "$(dirname "$0")/.."

MEMBER="$1"
EXPERIMENTS="$2"
GIT_REF="${3:-v1-experiments}"
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME to your Kaggle username}"

# One new private Kaggle script per launch. Names may only use lowercase letters, digits and dashes.
SLUG="group9-dqn-$(echo "$MEMBER" | tr '[:upper:]' '[:lower:]')-$(date +%m%d-%H%M)"
JOB_DIR="$(mktemp -d)"

sed -e "s/{{MEMBER}}/$MEMBER/" -e "s/{{EXPERIMENTS}}/$EXPERIMENTS/" -e "s/{{GIT_REF}}/$GIT_REF/" \
    kaggle/job.py > "$JOB_DIR/job.py"

cat > "$JOB_DIR/kernel-metadata.json" <<EOF
{
  "id": "$KAGGLE_USERNAME/$SLUG",
  "title": "$SLUG",
  "code_file": "job.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
EOF

uvx --with six kaggle kernels push -p "$JOB_DIR" --accelerator NvidiaTeslaT4
echo
echo "Job:              $SLUG ($MEMBER, experiments $EXPERIMENTS, code at $GIT_REF)"
echo "Check progress:   uvx --with six kaggle kernels status $KAGGLE_USERNAME/$SLUG"
echo "Download results: bash kaggle/fetch.sh $SLUG"
