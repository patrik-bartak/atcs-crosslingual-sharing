import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data["jaccard_sim"]["all_params"]


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/mask_sim",
        help="Path to the JSON outputs generated by 'get_masks.sh'.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        type=str,
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--task_name",
        default=SIB200,
        type=str,
        help="Name of the task.",
    )
    parser.add_argument(
        "--seeds",
        default=[41, 42, 43],
        type=int,
        nargs="+",
        help="Seeds to use.",
    )

    return parser.parse_args()


def get_lang_list(task):

    if task == SIB200:
        langs = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]
    
    elif task == XNLI:
        langs = ["cs", "hi", "id", "nl", "zh"]
    
    elif task == WIKIANN:
        langs = ["cs", "hi", "id", "nl", "zh"]
    
    elif task == TOXI:
        langs = ["hi", "zh-cn", "cs", "nl", "id"]
   
    else:
        return NotImplemented

    return langs


if __name__ == "__main__":

    args = argparser()
    langs = get_lang_list(args.task_name)

    # Initialize arrays for mean and std
    mean_matrix = np.zeros((len(langs), len(langs)))
    std_matrix = np.zeros((len(langs), len(langs)))

    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):

            if i < j:

                similarity_values = []
                for seed in args.seeds:

                    filename = os.path.join(
                        args.base_dir, f"{lang1}-{lang2}-{seed}.json"
                    )
                    if os.path.exists(filename):
                        similarity_values.append(read_json(filename))

                if similarity_values:
                    mean_similarity = np.mean(similarity_values)
                    std_similarity = np.std(similarity_values)
                    mean_matrix[i, j] = mean_similarity
                    std_matrix[i, j] = std_similarity

                else:
                    mean_matrix[i, j] = np.nan
                    std_matrix[i, j] = np.nan

    # Plotting

    # To check if it's a specific task (due to the backslash)
    if args.task_name == SIB200:
        task_name = "SIB200"

    else:
        task_name = args.task_name

    plt.figure(figsize=(6, 5))
    plt.imshow(mean_matrix, cmap="viridis", interpolation="nearest")
    cbar = plt.colorbar(label="Jaccard Similarity (Mean)")
    plt.xticks(range(len(langs)), langs, rotation=45, fontsize=12)
    plt.yticks(range(len(langs)), langs, fontsize=12)
    plt.title(
        f"Jaccard Similarity Matrix for {task_name}",
        weight="bold",
        fontsize=16,
    )
    plt.tight_layout()

    # Add mean and std to each cell
    for i in range(len(langs)):
        for j in range(len(langs)):

            if i < j and not np.isnan(mean_matrix[i, j]):
                plt.text(
                    j,
                    i,
                    f"{mean_matrix[i, j]:.2f}\n±{std_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

    savedir = f"{args.output_dir}/sim-{task_name}.png"
    plt.savefig(savedir, dpi=300)

    print(f"Saved to {savedir}!")
