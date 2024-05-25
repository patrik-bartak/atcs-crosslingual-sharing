import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *
from transformers import AutoModel

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
    parser.add_argument(
        "--exclude_others",
        action="store_true",
        help="Whether to exclude non-weight parameters.",
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


def plot_sparsity_per_layer(sparsity_dict, layers, output_path, task_name):
    languages = list(sparsity_dict.keys())
    layer_sparsity = {layer: [] for layer in layers}

    for lang in languages:
        for layer in layers:
            layer_name = f"encoder.layer.{layer}"
            sparsities = [
                sparsity
                for param, sparsity in sparsity_dict[lang].items()
                if layer_name in param
            ]
            average_sparsity = np.mean(sparsities) if sparsities else 0
            layer_sparsity[layer].append(average_sparsity)

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

    for lang in langs:
        masks_list = []
        for seed in args.seeds:
            model_path = os.path.join(args.base_dir, f"snip-{lang}-{seed}")
            if os.path.exists(model_path):
                model = AutoModel.from_pretrained(model_path)
                masks = extract_masks(model, args.exclude_others)
                masks_list.append(masks)

        if masks_list:
            sparsity_dict[lang] = average_sparsity(masks_list)

    # To check if it's a specific task (due to the backslash)
    if args.task_name == SIB200:
        task_name = "SIB200"

    else:
        task_name = args.task_name

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"sparsity_{task_name}.png")
    plot_sparsity_per_layer(sparsity_dict, layers, output_path, task_name)
