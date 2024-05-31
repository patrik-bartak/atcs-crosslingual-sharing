import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *
from utils.languages import get_lang_list


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data


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
        default="figures_replot_patrik",
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


if __name__ == "__main__":

    args = argparser()
    langs = get_lang_list(args.task_name)

    # Initialize arrays for mean and std
    mean_matrix = np.zeros((len(langs), len(langs)))
    std_matrix = np.zeros((len(langs), len(langs)))

    seeds = args.seeds

    for i, lang in enumerate(langs):
        similarity_values = []
        for k in range(len(seeds)):
            seed1 = seeds[k]
            seed2 = seeds[(k + 1) % len(seeds)]

            filename = os.path.join(
                args.base_dir, f"{lang}-{seed1}-{seed2}.json"
            )
            if os.path.exists(filename):
                data = read_json(filename)
                data = data["jaccard_sim_norm"]["encoder"]["normalized_jacc"]
                # data = data["jaccard_sim_norm"]["encoder"]["regular_jacc"]
                similarity_values.append(data)

        if similarity_values:
            mean_similarity = np.mean(similarity_values)
            std_similarity = np.std(similarity_values)
            mean_matrix[i, i] = mean_similarity
            std_matrix[i, i] = std_similarity

        else:
            mean_matrix[i, i] = np.nan
            std_matrix[i, i] = np.nan

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
        f"Stability Matrix for {task_name}",
        weight="bold",
        fontsize=16,
    )
    plt.tight_layout()

    # Add mean and std to each cell
    for i in range(len(langs)):
        for j in range(len(langs)):

            if i <= j and not np.isnan(mean_matrix[i, j]):
                plt.text(
                    j,
                    i,
                    f"{mean_matrix[i, j]:.3f}\n±{std_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                )

    savedir = f"{args.output_dir}/stability-{task_name}.png"
    plt.savefig(savedir, dpi=300)

    print(f"Saved to {savedir}!")
