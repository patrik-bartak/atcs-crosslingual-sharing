import argparse
import os

from transformers import AutoModel, AutoModelForSequenceClassification
import json

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
args = parser.parse_args()


def extract_masks(model):
    named_mask = {}
    for name, param in model.named_parameters():
        mask = param > 0
        named_mask[name] = mask
    return named_mask


def compute_union_intersect(mask_a, mask_b) -> tuple[int, int]:
    union = mask_a | mask_b
    intersect = mask_a & mask_b
    return intersect.sum().item(), union.sum().item()


def jaccard_each(masks_a, masks_b):
    """
    Compute jaccard similarity from dicts mapping model params to param masks.
    :param masks_a: Mask A.
    :param masks_b: Mask B.
    :return: Dict with jaccard similarity for each param group and also a
    similarity for all the params under "all_params".
    """
    union_params = set(masks_a.keys()) | set(masks_b.keys())
    jacc_sim_dict = {}

    union_count = 0
    overlap_count = 0
    for param in union_params:
        try:
            mask_a = masks_a[param]
            mask_b = masks_b[param]
        except KeyError:
            print(f"Cannot find param {param} in other model")
            jacc_sim_dict[param] = None
            continue

        intersect, union = compute_union_intersect(mask_a, mask_b)
        # Jacc sim
        jacc_sim_dict[param] = 0 if union == 0 else intersect / union
        union_count += union
        overlap_count += intersect

    jacc_sim_dict["all_params"] = 0 if union_count == 0 else overlap_count / union_count

    return jacc_sim_dict


model_paths = [args.model_a, args.model_b]
# model_paths = ["pruned_models/xnli/unpruned", "pruned_models/xnli/unpruned"]

model_mask_dicts = [
    extract_masks(AutoModel.from_pretrained(path, num_labels=3)) for path in model_paths
]

jacc_dict = jaccard_each(*model_mask_dicts)

os.makedirs(args.output_dir, exist_ok=True)
save_path = os.path.join(args.output_dir, "mask_sim.json")
with open(save_path, "w") as f:
    json.dump(jacc_dict, f, indent=4, sort_keys=True)
