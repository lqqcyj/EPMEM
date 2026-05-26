#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME=""
export TRANSFORMERS_CACHE=""

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"   # HF model ID or local path

LOG_DIR=""
RESULT_DIR=""
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# ── Hyper-parameter grid ─────────────────────────────────────────────────────

RETRACING_RATIOS=(0.1)
BETA_MASKS=(0.1 0.3)


# ── Fixed parameters ─────────────────────────────────────────────────────────
ENTROPY_THRESHOLD=0.7
BETA_CONF=0.12
MAX_NEW_TOKENS=256
LAYER=("11")

IMAGE_FOLDER=""

STAGE1_QUESTION=""


# stage_2 question file is written dynamically by model_vqa_loader_qwen.py
# (stage1_to_2), so the path below must be writable:
STAGE2_QUESTION_TEMPLATE="${RESULT_DIR}/stage2_question_PLACEHOLDER.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#RETRACING_RATIOS[@]} * ${#LAYER[@]} * ${#BETA_MASKS[@]} ))
COUNTER=0

EXPERIMENT_START=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$RESULT_DIR/summary_${EXPERIMENT_START}.txt"


echo "beta_masks=${BETA_MASKS[*]}  beta_conf=${BETA_CONF}  entropy_threshold=${ENTROPY_THRESHOLD}" \
                                                 | tee -a "$SUMMARY_FILE"
Choice: <choice>" | tee -a "$SUMMARY_FILE"

for RATIO in "${RETRACING_RATIOS[@]}"; do
    for LAYER_STR in "${LAYER[@]}"; do
        for MASK in "${BETA_MASKS[@]}"; do
            COUNTER=$(( COUNTER + 1 ))
            LAYER_SAFE=$(echo "$LAYER_STR" | tr ' ' '_')
            CONFIG="ratio${RATIO}_layers${LAYER_SAFE}_mask${MASK}"
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

            LOG_FILE="$LOG_DIR/${CONFIG}_${TIMESTAMP}.log"
            STAGE1_ANS="$RESULT_DIR/stage1_${CONFIG}_${TIMESTAMP}.jsonl"
            STAGE2_ANS="$RESULT_DIR/stage2_${CONFIG}_${TIMESTAMP}.jsonl"
            # stage2 question file: written by stage1_to_2() inside the python script
            STAGE2_Q="$RESULT_DIR/stage2_q_${CONFIG}_${TIMESTAMP}.jsonl"

            echo "" | tee -a "$SUMMARY_FILE"
            echo "[$COUNTER/$TOTAL] retracing_ratio=${RATIO}  trigger_layers=${LAYER_STR}  beta_mask=${MASK}" \
                | tee -a "$SUMMARY_FILE"
            echo "  start: $(date)" | tee -a "$SUMMARY_FILE"

            # ── Run two-stage evaluation ─────────────────────────────────────────
            python -m llava.eval.model_vqa_loader_qwen \
                --model-path "$MODEL_PATH" \
                --image-folder "$IMAGE_FOLDER" \
                --stage1_question_file  "$STAGE1_QUESTION" \
                --stage1_answers_file   "$STAGE1_ANS" \
                --stage2_question_file  "$STAGE2_Q" \
                --stage2_answers_file   "$STAGE2_ANS" \
                --temperature 0 \
                --max_new_tokens $MAX_NEW_TOKENS \
                --beta_mask  $MASK \
                --beta_conf  $BETA_CONF \
                --use_efvr \
                --efvr_entropy_threshold $ENTROPY_THRESHOLD \
                --efvr_retracing_ratio   $RATIO \
                --efvr_starting_layer    11 \
                --efvr_ending_layer      20 \
                --efvr_topk              6 \
                --efvr_trigger_layers    $LAYER_STR \
                2>&1 | tee "$LOG_FILE"

            # ── Compute accuracy ─────────────────────────────────────────────────
            echo "ACC ..." | tee -a "$SUMMARY_FILE"
            ACC_OUTPUT=$(python3 -m llava.eval.eval_emoset \
                --result-file "$STAGE2_ANS" \
                --output-dir  "$RESULT_DIR" \
                2>&1 | tee -a "$LOG_FILE")
            echo "$ACC_OUTPUT" | tee -a "$SUMMARY_FILE"
            echo "  end: $(date)" | tee -a "$SUMMARY_FILE"
        done
    done
done

