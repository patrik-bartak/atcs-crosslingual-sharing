#!/bin/bash

# Base directories and fixed first model path
BASE_DIR="pruned_models"
SEEDS=(41 42 43)
OUTPUT_DIR="results/mask_sim_snip/toxitext"
PRUNING_TYPE="snip" # or magnitude
#DATASET="xnli"
#DATASET="wikiann"
#DATASET="sib200"
DATASET="toxitext"

# Languages
#LANGUAGES=("cs" "hi" "id" "nl" "zh" "en") # Adjust this based on the languages you have
LANGUAGES=("cs" "hi" "id" "nl" "zh") # Adjust this based on the languages you have
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

NUM_SEEDS=${#SEEDS[@]}

# Loop through each language and seed combination
for ((i = 0; i < ${#LANGUAGES[@]}; i++)); do
    for ((k = 0; k < NUM_SEEDS; k++)); do
        SEED1=${SEEDS[k % NUM_SEEDS]}
        SEED2=${SEEDS[(k + 1) % NUM_SEEDS]}
        FIRST_LANG=${LANGUAGES[i]}
        FIRST_MODEL="${BASE_DIR}/${DATASET}/${PRUNING_TYPE}-${FIRST_LANG}-${SEED1}${EXTRA}"
        SECOND_MODEL="${BASE_DIR}/${DATASET}/${PRUNING_TYPE}-${FIRST_LANG}-${SEED2}${EXTRA}"

        # Define output name
        OUTPUT_NAME_1="${FIRST_LANG}-${SEED1}-${SEED2}.json"

        echo "Running for model: $FIRST_MODEL and $SECOND_MODEL..."
        python mask_similarity.py "$FIRST_MODEL" "$SECOND_MODEL" --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME_1"
    done
done
