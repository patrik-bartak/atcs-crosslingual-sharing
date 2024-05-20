#!/bin/bash

# Base directories and fixed first model path
BASE_DIR="<insert_model_main_dir_here>"
FIRST_MODEL="${BASE_DIR}/<insert_model_name_here>"
SECOND_MODEL_BASE="${BASE_DIR}/magnitude"
OUTPUT_DIR="results/mask_sim"

# Languages and seeds
LANGUAGES=("ces_Latn" "hin_Deva" "ind_Latn" "nld_Latn" "zho_Hans") # Adjust this based on the languages you have
SEEDS=(41 42 43)

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Loop through each language and seed combination
for LANG in "${LANGUAGES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        SECOND_MODEL="${SECOND_MODEL_BASE}/magnitude-${LANG}-${SEED}"
        OUTPUT_NAME="mag-${LANG}-${SEED}.json"
        
        echo "Running for model: $SECOND_MODEL, output: $OUTPUT_NAME"
        
        python mask_similarity.py "$FIRST_MODEL" "$SECOND_MODEL" --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME"
    done
done
