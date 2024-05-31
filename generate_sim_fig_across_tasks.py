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
        default="results/mask_sim_snip",
        help="Path to the JSON outputs generated by 'get_masks.sh'.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        type=str,
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--lang_name",
        default="en",
        type=str,
        help="2 digit code of the lang.",
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
    tasks = ["xnli", "wikiann", "sib200", "toxitext"]

    lang = args.lang_name

    # or regular_jacc
    for jacc_type in ["normalized_jacc", "regular_jacc"]:

        # Initialize arrays for mean and std
        mean_matrix = np.zeros((len(tasks), len(tasks)))
        std_matrix = np.zeros((len(tasks), len(tasks)))

        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):

                if i < j:

                    similarity_values = []
                    for seed in args.seeds:

                        filename = os.path.join(
                            args.base_dir, lang, f"{task1}-{task2}-{seed}.json"
                        )

                        if os.path.exists(filename):
                            data = read_json(filename)
                            data = data["jaccard_sim_norm"]["encoder"][jacc_type]
                            similarity_values.append(data)

                    if similarity_values:
                        mean_similarity = np.mean(similarity_values)
                        std_similarity = np.std(similarity_values)
                        mean_matrix[i, j] = mean_similarity
                        std_matrix[i, j] = std_similarity

                    else:
                        mean_matrix[i, j] = np.nan
                        std_matrix[i, j] = np.nan

        # Plotting

        plt.figure(figsize=(6, 5))
        plt.imshow(mean_matrix, cmap="viridis", interpolation="nearest")
        cbar = plt.colorbar(label="Jaccard Similarity (Mean)")
        plt.xticks(range(len(tasks)), tasks, rotation=45, fontsize=12)
        plt.yticks(range(len(tasks)), tasks, fontsize=12)
        plt.title(
            f"{'Normalized' if jacc_type == 'normalized_jacc' else ''} Jaccard Similarity Matrix for {lang}",
            weight="bold",
            fontsize=16,
        )
        plt.tight_layout()

        # Add mean and std to each cell
        for i in range(len(tasks)):
            for j in range(len(tasks)):

                if i < j and not np.isnan(mean_matrix[i, j]):
                    plt.text(
                        j,
                        i,
                        f"{mean_matrix[i, j]:.3f}\n±{std_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        savedir = f"{args.output_dir}/{jacc_type}-similarity-{lang}-across-tasks.png"
        plt.savefig(savedir, dpi=300)

        print(f"Saved to {savedir}!")
