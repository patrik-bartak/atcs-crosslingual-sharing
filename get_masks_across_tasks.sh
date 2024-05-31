#!/bin/bash

# Base directories and fixed first model path
BASE_DIR="pruned_models"
SEEDS=(41 42 43)
PRUNING_TYPE="snip" # or magnitude
#DATASET="xnli"
#DATASET="wikiann"
#DATASET="sib200"
DATASETS=("xnli" "wikiann" "sib200" "toxitext")

# "en", "nl", etc.
LANG=$1

# Languages
#LANGUAGES=("cs" "hi" "id" "nl" "zh" "en") # Adjust this based on the languages you have
OUTPUT_DIR="results/mask_sim_snip/$LANG"
#LANGUAGES=("cs" "hi" "id" "nl" "zh") # Adjust this based on the languages you have
#LANGUAGES=("ces_Latn" "hin_Deva" "ind_Latn" "nld_Latn" "zho_Hans") # Adjust this based on the languages you have

#EXTRA="/model.pt"
EXTRA=""

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
    for ((i = 0; i < ${#DATASETS[@]}; i++)); do
        FIRST_DSET="${DATASETS[i]}"
        FIRST_MODEL="${BASE_DIR}/${FIRST_DSET}/${PRUNING_TYPE}-${LANG}-${SEED}${EXTRA}"
        
        # Loop through the rest of the languages
        for ((j = i + 1; j < ${#DATASETS[@]}; j++)); do
            SECOND_DSET="${DATASETS[j]}"
            SECOND_MODEL="${BASE_DIR}/${SECOND_DSET}/${PRUNING_TYPE}-${LANG}-${SEED}${EXTRA}"
            
            # Define output names for both possible combinations
            OUTPUT_NAME_1="${FIRST_DSET}-${SECOND_DSET}-${SEED}.json"
            OUTPUT_NAME_2="${SECOND_DSET}-${FIRST_DSET}-${SEED}.json"
            
            # Check if either output file already exists
            if output_file_exists "$OUTPUT_NAME_1" || output_file_exists "$OUTPUT_NAME_2"; then
                echo "Comparison already made for $FIRST_DSET and $SECOND_DSET for seed $SEED. Skipping..."
            else
                echo "Running for model: $FIRST_MODEL and $SECOND_MODEL..."
                python mask_similarity.py "$FIRST_MODEL" "$SECOND_MODEL" --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME_1"
            fi
        done
    done
done
