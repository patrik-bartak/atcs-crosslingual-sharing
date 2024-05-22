import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from utils.constants import *


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


def get_lang_list(task):
    if task == SIB200:
        langs = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]

    elif task == XNLI:
        langs = ["cs", "hi", "id", "nl", "zh"]

    else:
        return NotImplemented

    return langs


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # python plotting/plot_per_layer.py --task_name=xnli --json_dir=results_snellius/mask_sim
    args = argparser()
    langs = get_lang_list(args.task_name)
    layers = list(range(12))  # Assuming 12 layers for XLM-R

    sparsity_dict = {lang: {} for lang in langs}
    lang_pair_to_similarities_dict = dict()

    for i in range(5):
        for j in range(i + 1, 5):
            lang1 = langs[i]
            lang2 = langs[j]
            lang_pair_similarities = []
            for seed in args.seeds:
                json_path = os.path.join(args.json_dir, f"{lang1}-{lang2}-{seed}.json")
                sim_data = read_json(json_path)["jaccard_sim"]
                # TODO: finish by selecting the key in the json dict containing the similarity for the entire layer
                #       we cannot simply take the mean because the parts within the layer have different number of params
                layer_sim_data = {k: v for k, v in sim_data.items() if "layer" in k}
                lang_pair_similarities.append(...)

            lang_pair_to_similarities_dict[f"{lang1}-{lang2}"] = np.mean(lang_pair_similarities, axis=0)

    print(lang_pair_to_similarities_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"similarity_{args.task_name}.png")

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
        f"Mean Similarity per Layer for {args.task_name}", fontsize=16, fontweight="bold"
    )
    ax.legend()

    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    # plt.savefig(output_path, dpi=300)
    # print(f"Saved to sparsity_{args.task_name}.png!")
    plt.show()
