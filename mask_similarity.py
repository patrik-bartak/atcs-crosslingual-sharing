import os
import json
import argparse

import numpy as np
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
        "--not_exclude_others",
        action="store_false",
        help="Whether to exclude non-weight parameters.",
    )

    return parser.parse_args()


# Function to group tensors by a specific substring
def group_tensors_by_substring(tensor_dict, substring):
    grouped_tensors = {}

    tensors_a = []
    tensors_b = []

    for key, (tensor_a, tensor_b) in tensor_dict.items():
        if substring in key:
            tensors_a.append(tensor_a)
            tensors_b.append(tensor_b)
    grouped_tensors[substring] = (tensors_a, tensors_b)
    return grouped_tensors


def extract_masks(model, exclude=True):
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
    """
    Sparsity is the ratio of zero elements to total elements.
    :param mask:
    :return:
    """
    return (mask == 0).sum().item() / mask.numel()


def get_param_to_masks_dict(union_params, masks_a, masks_b):
    d = {}
    for param in union_params:
        try:
            mask_a = masks_a[param]
            mask_b = masks_b[param]
            d[param] = (mask_a, mask_b)
        except KeyError:
            print(f"Cannot find param {param} in other model")
            # d[param] = None
            continue
    return d


def get_zero_and_total_from_list(tensors: list):
    zero_count = sum((tensor == 0).sum().item() for tensor in tensors)
    total_count = sum(tensor.numel() for tensor in tensors)
    return zero_count, total_count


def compute_sparsity_metric(param_to_masks, path_a, path_b):
    sparsity_dict = {}
    # For all params sparsity
    zero_count_a = 0
    total_count_a = 0
    zero_count_b = 0
    total_count_b = 0

    for param, (masks_a, masks_b) in param_to_masks.items():
        assert len(masks_a) != 0, param
        assert len(masks_b) != 0, param
        # Computing sparsity for each parameter group
        param_zero_count_a, param_total_count_a = get_zero_and_total_from_list(masks_a)
        param_zero_count_b, param_total_count_b = get_zero_and_total_from_list(masks_b)

        # Update overall zero counts and total counts
        zero_count_a += param_zero_count_a
        total_count_a += param_total_count_a
        zero_count_b += param_zero_count_b
        total_count_b += param_total_count_b

        # Calculate sparsity for current parameter group
        sparsity_a = param_zero_count_a / param_total_count_a
        sparsity_b = param_zero_count_b / param_total_count_b
        sparsity_dict[param] = {path_a: sparsity_a, path_b: sparsity_b}

    # No longer necessary
    # sparsity_dict["all_params"] = {
    #     path_a: zero_count_a / total_count_a,
    #     path_b: zero_count_b / total_count_b,
    # }
    return sparsity_dict


def compute_intersect_union(mask_a, mask_b):
    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return intersection, union


def compute_jaccard(mask_a, mask_b):
    intersection, union = compute_intersect_union(mask_a, mask_b)
    return 0 if union == 0 else intersection / union


def get_intersect_union_from_list(masks_a, masks_b):
    intersect_count = sum((a & b).sum().item() for a, b in zip(masks_a, masks_b))
    union_count = sum((a | b).sum().item() for a, b in zip(masks_a, masks_b))
    return intersect_count, union_count


def compute_jaccard_metric(param_to_masks, path_a, path_b):
    jacc_dict = {}
    # For all params jaccard similarity
    union_count = 0
    intersect_count = 0

    for param, (masks_a, masks_b) in param_to_masks.items():
        # Computing Jaccard similarity for each parameter group
        intersect, union = get_intersect_union_from_list(masks_a, masks_b)
        jacc_dict[param] = 0 if union == 0 else intersect / union

        # Update overall intersection and union counts
        intersect_count += intersect
        union_count += union

    # No longer necessary
    # jacc_dict["all_params"] = 0 if union_count == 0 else intersect_count / union_count
    return jacc_dict


def get_rand_mask_of_sparsity(shape, sparsity):
    random_tensor = torch.rand(shape)
    return random_tensor >= sparsity


from scipy import stats


def compute_jaccard_metric(param_to_masks, path_a, path_b):
    jacc_dict = {}
    # For all params jaccard similarity
    union_count = 0
    intersect_count = 0

    for param, (masks_a, masks_b) in param_to_masks.items():
        # Computing Jaccard similarity for each parameter group
        intersect, union = get_intersect_union_from_list(masks_a, masks_b)
        jacc_dict[param] = 0 if union == 0 else intersect / union

        # Update overall intersection and union counts
        intersect_count += intersect
        union_count += union

    # No longer necessary
    # jacc_dict["all_params"] = 0 if union_count == 0 else intersect_count / union_count
    return jacc_dict


def compute_jaccard_normalized_metric(param_to_masks, path_a, path_b):
    jacc_norm_dict = {}
    # For all params jaccard sim
    union_count = 0
    intersect_count = 0
    for param, (masks_a, masks_b) in param_to_masks.items():
        param_u_c = 0
        param_i_c = 0
        random_int_uns = []
        for mask_a, mask_b in zip(masks_a, masks_b):
            spars_a = compute_mask_sparsity(mask_a)
            spars_b = compute_mask_sparsity(mask_b)
            # Pairs of random masks of equivalent sparsity
            rand_masks = [(get_rand_mask_of_sparsity(mask_a.shape, spars_a),
                           get_rand_mask_of_sparsity(mask_b.shape, spars_b)) for _ in range(10)]
            # Generate baseline Jaccard distribution from random masks of equivalent sparsity
            random_int_uns.append([compute_intersect_union(rand_a, rand_b) for rand_a, rand_b in rand_masks])
            # [
            #     [i u, i u, i u, i u, i u, i u, i u, i u, i u], w
            #     [i u, i u, i u, i u, i u, i u, i u, i u, i u], b
            #     [i u, i u, i u, i u, i u, i u, i u, i u, i u], k
            #     [i u, i u, i u, i u, i u, i u, i u, i u, i u], q
            #     [i u, i u, i u, i u, i u, i u, i u, i u, i u], v
            # ]
        i_s, u_s = get_intersect_union_from_list(masks_a, masks_b)
        param_u_c += u_s
        param_i_c += i_s

        random_int_uns_t = zip(*random_int_uns)
        rand_jaccs = []
        for random_int_un in random_int_uns_t:
            # i_u_5s
            rand_jacc = sum([i for i, u in random_int_un]) / sum([u for i, u in random_int_un])
            rand_jaccs.append(rand_jacc)

        # Get the mean and z score
        observed_similarity = 0 if param_u_c == 0 else param_i_c / param_u_c
        mean_random_similarity = np.mean(rand_jaccs)
        std_random_similarity = np.std(rand_jaccs)

        z_score = (observed_similarity - mean_random_similarity) / std_random_similarity
        p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed

        # Computing jaccard sim
        jacc_norm_dict[param] = {
            "regular_jacc": observed_similarity,
            "normalized_jacc": observed_similarity - mean_random_similarity,
            "mean_random_jacc": mean_random_similarity,
            "std_random_jacc": std_random_similarity,
            "z_score": z_score,
            "p_value": p_value,
        }

        union_count += param_u_c
        intersect_count += param_i_c

    # No longer necessary
    # jacc_norm_dict["all_params"] = 0 if union_count == 0 else intersect_count / union_count
    return jacc_norm_dict


def count_elements(tensors: list):
    total = 0
    for tensor in tensors:
        total += tensor.numel()
    return total


def compute_params_count_metric(param_to_masks, path_a, path_b):
    num_params_dict = {}
    # For all params count
    params_count_a = 0
    params_count_b = 0
    for param, (masks_a, masks_b) in param_to_masks.items():
        # Computing number of params
        num_el_a = count_elements(masks_a)
        num_el_b = count_elements(masks_b)
        num_params_dict[param] = {path_a: num_el_a, path_b: num_el_b}
        params_count_a += num_el_a
        params_count_b += num_el_b

    # No longer necessary
    # num_params_dict["all_params"] = {
    #     path_a: params_count_a,
    #     path_b: params_count_b,
    # }
    return num_params_dict


def compute_model_metrics(masks_a, masks_b, path_a, path_b):
    """
    Compute jaccard similarity from dicts mapping model params to param masks.
    :param path_b:
    :param path_a:
    :param masks_a: Mask A.
    :param masks_b: Mask B.
    :return: Dict with jaccard similarity for each param group and also a
    similarity for all the params under "all_params".
    """
    union_params = set(masks_a.keys()) | set(masks_b.keys())
    param_to_masks = get_param_to_masks_dict(union_params, masks_a, masks_b)
    # Group the parameters
    substrings = [*[f"encoder.layer.{i}." for i in range(12)], "encoder"]
    param_to_masks_grouped = {}
    [param_to_masks_grouped.update(group_tensors_by_substring(param_to_masks, substring)) for substring in substrings]

    return {
        "jaccard_sim_norm": compute_jaccard_normalized_metric(param_to_masks_grouped, path_a, path_b),
        "jaccard_sim": compute_jaccard_metric(param_to_masks_grouped, path_a, path_b),
        "sparsity": compute_sparsity_metric(param_to_masks_grouped, path_a, path_b),
        "num_params": compute_params_count_metric(param_to_masks_grouped, path_a, path_b),
    }


# import torch
# import torch.nn as nn
#
# def custom_initialize_weights(module):
#     sparsity = 0.45
#     # Probability of getting a 1
#     prob_thres = 1 - sparsity
#
#     if isinstance(module, (nn.Linear, nn.Embedding)):
#         with torch.no_grad():
#             prob = torch.rand(module.weight.size())
#             module.weight.copy_(torch.where(prob < prob_thres, torch.ones_like(prob), torch.zeros_like(prob)))
#
#     if hasattr(module, "bias") and module.bias is not None:
#         with torch.no_grad():
#             prob = torch.rand(module.bias.size())
#             module.bias.copy_(torch.where(prob < prob_thres, torch.ones_like(prob), torch.zeros_like(prob)))
#     if hasattr(module, "weight") and module.weight is not None:
#         with torch.no_grad():
#             prob = torch.rand(module.weight.size())
#             module.weight.copy_(torch.where(prob < prob_thres, torch.ones_like(prob), torch.zeros_like(prob)))
#
#     # Custom initialization for attention matrices
#     if hasattr(module, 'attention') and hasattr(module.attention, 'self'):
#         for param in ['query', 'key', 'value']:
#             attn_param = getattr(module.attention.self, param).weight
#             with torch.no_grad():
#                 prob = torch.rand(attn_param.size())
#                 attn_param.copy_(torch.where(prob < prob_thres, torch.ones_like(prob), torch.zeros_like(prob)))


if __name__ == "__main__":
    from transformers.utils import logging
    # To suppress warnings regarding task-specific layers being randomly initialized
    logging.set_verbosity_error()

    args = argparser()
    model_paths = [args.model_a, args.model_b]
    model_mask_dicts = [
        extract_masks(AutoModel.from_pretrained(path), args.not_exclude_others)
        for path in model_paths
    ]

    metric_dict = compute_model_metrics(*model_mask_dicts, *model_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4, sort_keys=True)
