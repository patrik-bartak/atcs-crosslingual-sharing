import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from generate_sim_fig_across_langs import read_json
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
        help="Base directory where the metric jsons are stored.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        type=str,
        help="Path to save the mask.",
    )

    return parser.parse_args()


def extract_masks(model, exclude=False):
    named_mask = {}
    excluded_params = [
        "pooler",
        "embeddings",
        "bias",
        "LayerNorm",
    ]  # Parameters that should not influence similarity
    for name, param in model.named_parameters():
        if any(exp in name for exp in excluded_params) and exclude:
            continue
        else:
            mask = param != 0
            named_mask[name] = mask
    return named_mask


def compute_mask_sparsity(mask):

    return (mask == 0).sum().item() / mask.numel()


def average_sparsity(masks_list):
    average_sparsity_dict = {}
    for masks in masks_list:
        for param, mask in masks.items():
            if param not in average_sparsity_dict:
                average_sparsity_dict[param] = []
            sparsity = compute_mask_sparsity(mask)
            average_sparsity_dict[param].append(sparsity)

    for param in average_sparsity_dict:
        average_sparsity_dict[param] = np.mean(average_sparsity_dict[param])

    return average_sparsity_dict


def plot_sparsity_per_layer(layer_sparsity, languages, output_path, task_name):
    # Plotting
    x = np.arange(len(layers))
    width = 0.15

    fig, ax = plt.subplots(figsize=(7, 4))

    pruning_type = "mag_pruning"

    for idx, lang in enumerate(languages):
        # label = f"{lang}+{pruning_type}"
        label = f"{lang}"

        ax.bar(
            x + idx * width,
            [layer_sparsity[layer][idx] for layer in layers],
            width,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Sparsity")
    ax.set_title(
        f"Mean Sparsity per Layer for {task_name}", fontsize=16, fontweight="bold"
    )
    ax.set_xticks(x + width * (len(languages) - 1) / 2)
    ax.set_xticklabels(layers)
    ax.legend()

    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved to sparsity_{task_name}.png!")


if __name__ == "__main__":
    args = argparser()
    langs = get_lang_list(args.task_name)
    layers = list(range(12))  # Assuming 12 layers
    seeds = args.seeds

    task_name = args.task_name

    layer_sparsity = np.zeros((len(layers), len(langs), len(seeds)))

    for idx, lang in enumerate(langs):
        masks_list = []
        for i in range(len(seeds)):
            seed1 = seeds[i]
            seed2 = seeds[(i + 1) % len(seeds)]
            metrics_path = os.path.join(args.json_dir, f"{lang}-{seed1}-{seed2}.json")
            if os.path.exists(metrics_path):
                data = read_json(metrics_path)
                spl = data["sparsity"]
                for l in layers:
                    layer_sparsity[l, idx, i] = list(spl[f"encoder.layer.{l}."].values())[0]

    layer_sparsity = layer_sparsity.mean(axis=2)
    print(layer_sparsity)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"sparsity_{task_name}.png")
    plot_sparsity_per_layer(layer_sparsity, langs, output_path, task_name)
