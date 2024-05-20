import argparse
import json
import os

import torch
from transformers import AutoModel


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_a",
        type=str,
        help="Path to first model.",
    )
    parser.add_argument(
        "model_b",
        type=str,
        help="Path to second model.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/mask_sim",
        type=str,
        help="Path to save the mask.",
    )
    parser.add_argument(
        "--output_name",
        default="mask_metrics.json",
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
            mask = torch.nonzero(param)
            named_mask[name] = mask

    return named_mask


def compute_union_intersect(mask_a, mask_b) -> tuple[int, int]:
    union = mask_a | mask_b
    intersect = mask_a & mask_b
    return intersect.sum().item(), union.sum().item()


def compute_mask_sparsity(mask):
    """
    Sparsity is the ratio of zero elements to total elements.
    :param mask:
    :return:
    """
    return (mask == 0).sum().item() / mask.numel()


def compute_sparsity_and_jaccard(masks_a, masks_b, path_a, path_b):
    """
    Compute jaccard similarity from dicts mapping model params to param masks.
    :param masks_a: Mask A.
    :param masks_b: Mask B.
    :return: Dict with jaccard similarity for each param group and also a
    similarity for all the params under "all_params".
    """
    union_params = set(masks_a.keys()) | set(masks_b.keys())
    jacc_dict = {}
    sparsity_dict = {}

    # For all params sparsity
    zero_count_a = 0
    total_count_a = 0
    zero_count_b = 0
    total_count_b = 0
    # For all params jaccard sim
    union_count = 0
    overlap_count = 0
    for param in union_params:
        try:
            mask_a = masks_a[param]
            mask_b = masks_b[param]

        except KeyError:
            print(f"Cannot find param {param} in other model")
            jacc_dict[param] = None
            continue
        # Computing sparsity for each model
        sparsity_a = compute_mask_sparsity(mask_a)
        sparsity_b = compute_mask_sparsity(mask_b)
        zero_count_a += (mask_a == 0).sum().item()
        total_count_a += mask_a.numel()
        zero_count_b += (mask_b == 0).sum().item()
        total_count_b += mask_b.numel()
        sparsity_dict[param] = {path_a: sparsity_a, path_b: sparsity_b}
        # Computing jaccard sim
        intersect, union = compute_union_intersect(mask_a, mask_b)
        jacc_dict[param] = 0 if union == 0 else intersect / union
        union_count += union
        overlap_count += intersect
    # Add all_params for sparsity and jaccard
    jacc_dict["all_params"] = 0 if union_count == 0 else overlap_count / union_count
    sparsity_dict["all_params"] = {
        path_a: zero_count_a / total_count_a,
        path_b: zero_count_a / total_count_a,
    }

    return {"jaccard_sim": jacc_dict, "sparsity": sparsity_dict}


if __name__ == "__main__":
    args = argparser()
    model_paths = [args.model_a, args.model_b]
    model_mask_dicts = [
        extract_masks(AutoModel.from_pretrained(path), args.exclude_others)
        for path in model_paths
    ]

    metric_dict = compute_sparsity_and_jaccard(*model_mask_dicts, *model_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4, sort_keys=True)
