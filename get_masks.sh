#!/bin/bash

# Base directories and fixed first model path
BASE_DIR="pruned_models"
SEEDS=(41 42 43)
OUTPUT_DIR="results/mask_sim_snip"

# Languages
LANGUAGES=("cs" "hi" "id" "nl" "zh") # Adjust this based on the languages you have

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check if the output file exists
output_file_exists() {
    local output_file="$OUTPUT_DIR/$1"
    [ -f "$output_file" ]
}

# Loop through each seed
for SEED in "${SEEDS[@]}"; do
    # Loop through each language and seed combination
    for ((i = 0; i < ${#LANGUAGES[@]}; i++)); do
        FIRST_LANG="${LANGUAGES[i]}"
        FIRST_MODEL="${BASE_DIR}/xnli/snip-${FIRST_LANG}-${SEED}"
        
        # Loop through the rest of the languages
        for ((j = i + 1; j < ${#LANGUAGES[@]}; j++)); do
            SECOND_LANG="${LANGUAGES[j]}"
            SECOND_MODEL="${BASE_DIR}/xnli/snip-${SECOND_LANG}-${SEED}"
            
            # Define output names for both possible combinations
            OUTPUT_NAME_1="${FIRST_LANG}-${SECOND_LANG}-${SEED}.json"
            OUTPUT_NAME_2="${SECOND_LANG}-${FIRST_LANG}-${SEED}.json"
            
            # Check if either output file already exists
            if output_file_exists "$OUTPUT_NAME_1" || output_file_exists "$OUTPUT_NAME_2"; then
                echo "Comparison already made for $FIRST_LANG and $SECOND_LANG for seed $SEED. Skipping..."
            else
                echo "Running for model: $FIRST_MODEL and $SECOND_MODEL..."
                python mask_similarity.py "$FIRST_MODEL" "$SECOND_MODEL" --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME_1" --exclude_others
            fi
        done
    done
done
