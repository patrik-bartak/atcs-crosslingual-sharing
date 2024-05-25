import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from generate_sim_fig import read_json
from utils.constants import *
from transformers import AutoModel


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
        "--base_dir",
        type=str,
        required=True,
        help="Base directory where the models are stored.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        type=str,
        help="Path to save the mask.",
    )

    return parser.parse_args()


def get_lang_list(task):

    if task == SIB200:
        langs = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]

    else:
        langs = ["cs", "hi", "id", "nl", "zh", "en"]

    return langs


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

    for idx, lang in enumerate(languages):
        ax.bar(
            x + idx * width,
            [layer_sparsity[layer][idx] for layer in layers],
            width,
            label=lang,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Sparsity")
    ax.set_title(
        f"Average Sparsity per Layer for {task_name}", fontsize=16, fontweight="bold"
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

    sparsity_dict = {lang: {} for lang in langs}
    seeds = args.seeds

    layer_sparsity = np.zeros((len(layers), len(langs), len(seeds)))

    for idx, lang in enumerate(langs):
        masks_list = []
        for i in range(len(seeds)):
            seed1 = seeds[i]
            seed2 = seeds[(i + 1) % len(seeds)]
            metrics_path = os.path.join(args.base_dir, f"{lang}-{seed1}-{seed2}.json")
            if os.path.exists(metrics_path):
                data = read_json(metrics_path)
                spl = data["sparsity"]
                for l in layers:
                    layer_sparsity[l, idx, i] = list(spl[f"encoder.layer.{l}."].values())[0]

    layer_sparsity = layer_sparsity.mean(axis=2)

    # To check if it's a specific task (due to the backslash)
    if args.task_name == SIB200:
        task_name = "SIB200"

    else:
        task_name = args.task_name

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"sparsity_{task_name}.png")
    plot_sparsity_per_layer(layer_sparsity, langs, output_path, task_name)
