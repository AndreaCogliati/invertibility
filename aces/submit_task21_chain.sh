#!/bin/bash
# Submit TASK-21 training as N sequential segments (never concurrent writers) with a given walltime.
#   bash aces/submit_task21_chain.sh <segments> <walltime HH:MM:SS> [first-dependency-jobid]
# Example for a 2-day partition limit: bash aces/submit_task21_chain.sh 2 48:00:00
# Segments use afterany, so a segment runs after the previous one ends for any reason; one that
# starts after the 48 h budget is spent exits at once. Prints the job IDs; submits nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:?number of segments}; WALL=${2:?walltime HH:MM:SS}; DEP=${3:-}
mkdir -p aces/logs
for i in $(seq 1 "$N"); do
    if [ -n "$DEP" ]; then
        JID=$(sbatch --parsable --time="$WALL" --dependency=afterany:"$DEP" aces/run_task21_segment.slurm)
    else
        JID=$(sbatch --parsable --time="$WALL" aces/run_task21_segment.slurm)
    fi
    echo "segment $i: job $JID (walltime $WALL${DEP:+, after $DEP})"
    DEP=$JID
done
echo "last segment: $DEP -- submit the evaluation after it finishes: sbatch aces/run_task21_eval.slurm"
