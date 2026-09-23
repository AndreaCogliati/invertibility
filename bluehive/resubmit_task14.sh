#!/bin/bash
# TASK-14 (tasks/TASK-14.md): resubmission helper for the Gate 5 full-GTZAN pilot.
#
# The training script (scripts/train_gate5_full_gtzan_wandb.py) auto-resumes whenever it's
# launched with --resume auto and runs/task14_gate5_full_gtzan/checkpoints/latest.pt already
# exists (bluehive/run_task14.slurm already passes --resume auto unconditionally) -- so
# resubmission is just "run sbatch again." This script exists to make that explicit, check that a
# checkpoint actually exists first (so a resubmission after the run never silently started a
# fresh run 2, which CLAUDE.md's "resuming must append to the same run history, not create
# ambiguous duplicate runs" rule forbids), and print the W&B run this resumption will continue.
#
# Usage: bluehive/resubmit_task14.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="runs/task14_gate5_full_gtzan/checkpoints/latest.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: no checkpoint found at $CHECKPOINT -- nothing to resume." >&2
    echo "Submit the fresh run with: sbatch bluehive/run_task14.slurm" >&2
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT"
uv run python -c "
import torch
state = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
print(f\"  step: {state['step']}\")
print(f\"  best_metric so far: {state.get('best_metric')} (step {state.get('best_step')})\")
print(f\"  git_commit at last checkpoint: {state.get('git_commit')}\")
print(f\"  full_gtzan_manifest_hash: {state['manifest_hash']}\")
"

echo
echo "Submitting bluehive/run_task14.slurm (already passes --resume auto) ..."
sbatch bluehive/run_task14.slurm

echo
echo "To find the W&B run this resumed job continues logging to, check the new job's .out file"
echo "for the 'wandb: Resuming run' or 'View run' line, or query the wandb_run_id field this"
echo "task's report.json/checkpoints record from the prior submission."
