#!/usr/bin/env bash

# ===========================
# Master Runner Script
# ===========================
# Usage:
#   sh master_run.sh <baseline|attack> <mitigation> <experiment_name>
#
# Example:
#   sh master_run.sh baseline normal chatdoctor-email
#
# ===========================

set -e

MODE="$1"            # baseline or attack
MIT="$2"             # summarizer, normal, personal_remove, instruct, remove
EXP="$3"             # experiment name

if [ $# -ne 3 ]; then
    echo "ERROR: Expected 3 arguments, got $#"
    echo "Usage: sh master_run.sh <baseline|attack> <mitigation> <experiment_name>"
    exit 1
fi

# Validate MODE
if [ "$MODE" != "baseline" ] && [ "$MODE" != "attack" ]; then
    echo "ERROR: Invalid MODE: $MODE"
    echo "Valid options: baseline | attack"
    exit 1
fi

# Validate MIT
if [ "$MIT" != "normal" ] && \
   [ "$MIT" != "summarizer" ] && \
   [ "$MIT" != "personal_remove" ] && \
   [ "$MIT" != "instruct" ] && \
   [ "$MIT" != "remove_cmd" ]; then
    echo "ERROR: Invalid mitigation type: $MIT"
    echo "Valid options: normal | summarizer | personal_remove | instruct | remove_cmd"
    exit 1
fi


echo "---------------------------------------"
echo "  MODE       : $MODE"
echo "  MITIGATION : $MIT"
echo "  EXPERIMENT : $EXP"
echo "---------------------------------------"


###################################
# Step 1: Run appropriate generator
###################################

echo "[STEP 1] Running prompt generator..."

GEN_CMD=""


case "$MIT" in
    normal)
        echo "Running NORMAL generation"
        GEN_CMD="python generate_prompt.py"
        ;;
    summarizer)
        echo "Running SUMMARIZER mitigation"
        GEN_CMD="python generate_prompt_summarizer.py"
	;;
    personal_remove)
        echo "Running PERSONAL REMOVE mitigation"
        GEN_CMD="python generate_prompt_sanitize.py"
        ;;
    instruct)
        echo "Running INSTRUCT mitigation"
        GEN_CMD="python generate_prompt_instruct.py"
        ;;
    remove_cmd)
        echo "Running REMOVE mitigation"
        GEN_CMD="python generate_prompt_cmd.py"
        ;;
    *)
        echo "Invalid mitigation: $MIT"
        exit 1
        ;;
esac

# Add --flag=1 only for baseline
if [ "$MODE" = "baseline" ]; then
    GEN_CMD="$GEN_CMD --flag=1"
fi

echo "Running: $GEN_CMD"
$GEN_CMD

###################################
# Step 2: Run experiment script
###################################
echo "---------------------------------------"
echo "[STEP 2] Executing experiment shell script..."

if [ ! -f "./$EXP.sh" ]; then
    echo "ERROR: Script $EXP.sh not found!"
    exit 1
fi

sh "./$EXP.sh"


###################################
# Step 3: Run correct evaluator
###################################
echo "---------------------------------------"
echo "[STEP 3] Running evaluation..."

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline evaluation"
    python evaluation_baseline.py --exp_name "$EXP"
else
    echo "Running attack evaluation"
    python evaluation_results.py --exp_name "$EXP"
fi

echo "---------------------------------------"
echo "Pipeline completed for experiment: $EXP with $MODE and $MIT"
echo "---------------------------------------"

