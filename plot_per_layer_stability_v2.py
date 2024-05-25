import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from utils.constants import *
from utils.languages import get_lang_list


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=SIB200,
        type=str,
        help="Name of the task.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43],
        help="List of seeds to process.",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Directory where the json results are stored.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        type=str,
        help="Path to save the plots.",
    )
    return parser.parse_args()


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # python plot_per_layer.py --task_name=xnli --json_dir=results/mask_sim_snip
    args = argparser()
    langs = get_lang_list(args.task_name)
    layers = list(range(12))  # Assuming 12 layers for XLM-R

    sparsity_dict = {lang: {} for lang in langs}
    lang_pair_to_similarities_dict = dict()

    seeds = args.seeds

    for i in range(len(langs)):
        lang_pair_similarities = []
        lang = langs[i]
        for j in range(len(seeds)):
            seed1 = seeds[j]
            seed2 = seeds[(j + 1) % len(seeds)]

            json_path = os.path.join(args.json_dir, f"{lang}-{seed1}-{seed2}.json")
            sim_norm_data = read_json(json_path)["jaccard_sim_norm"]
            # Assuming that the results dict contains layer.i keys for ith layer jaccard similarities
            layer_sim_data = [sim_norm_data[f"encoder.layer.{l}."]["normalized_jacc"] for l in layers]
            if len(layer_sim_data) == 0:
                raise Exception(
                    "No fields like 'layer.1', 'layer.2', etc found in file. Rerun the mask similarity script to get those."
                )
            lang_pair_similarities.append(layer_sim_data)

        lang_pair_to_similarities_dict[f"{lang}"] = np.mean(
            lang_pair_similarities, axis=0
        )

    print(lang_pair_to_similarities_dict)

    # To check if it's a specific task (due to the backslash)
    if args.task_name == SIB200:
        task_name = "SIB200"

    else:
        task_name = args.task_name

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"stability_layer_{task_name}.png")

    fig, ax = plt.subplots(figsize=(7, 4))

    for lang_pair, data in lang_pair_to_similarities_dict.items():
        ax.plot(
            layers,
            data,
            label=lang_pair,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Similarity")
    ax.set_title(
        f"Mean Similarity per Layer for {task_name}",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend()
    # ax.set_ylim([0, 1])

    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Saved to {output_path}")
    plt.show()
