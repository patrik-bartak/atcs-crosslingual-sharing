import re
from utils.constants import *
from prune import build_model_tokenizer
from parsing import eval_parser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300

_EXCLUDED_LAYERS = [
    "roberta.pooler.dense.weight",
    "roberta.pooler.dense.bias",
    "classifier.weight",
    "classifier.bias",
    "classifier.dense.weight",
    "classifier.dense.bias",
    "classifier.out_proj.weight",
    "classifier.out_proj.bias",
]


def extract_mask(model, filters):
    named_mask = {}
    for name, param in model.named_parameters():
        if name in filters:
            continue
        mask = param > 0
        named_mask[name] = mask
    return named_mask


def jaccard_all_params_from(masks_a, masks_b):
    # assume masks_a and masks_b contains the same keys and corresponding
    # tensor dimension
    union_count = 0
    overlap_count = 0
    for name_param, mask_a in masks_a.items():
        mask_b = masks_b[name_param]
        union = mask_a | mask_b
        union_count += union.sum().item()
        overlap = mask_a & mask_b
        overlap_count += overlap.sum().item()
    return overlap_count / union_count


def extract_num(sample_string):
    numbers = re.findall(r"\d+", sample_string)
    numbers = list(map(int, numbers))
    return numbers[0]


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def jaccard_layer_from(masks_a, masks_b):
    # assume masks_a and masks_b contains the same keys and corresponding
    # tensor dimenstion
    layer_overlap = {}
    for name_param, mask_a in masks_a.items():
        if not has_numbers(name_param):
            continue
        l_n = extract_num(name_param)
        mask_b = masks_b[name_param]
        union = mask_a | mask_b
        union_count = union.sum().item()
        overlap = mask_a & mask_b
        overlap_count = overlap.sum().item()
        if l_n in layer_overlap:
            layer_overlap[l_n][0] += overlap_count
            layer_overlap[l_n][1] += union_count
        layer_overlap[l_n] = [overlap_count, union_count]
    return layer_overlap


def plot_jaccard_all(datasets, heatmap, lang, save_path):
    # Labels for x and y axes
    x_labels = datasets
    y_labels = datasets

    # Plotting the heatmap
    mean_mat = np.mean(heatmap, axis=2)
    std_mat = np.std(heatmap, axis=2)

    # Format mean ± std for each cell
    annot = np.array(
        [
            [
                "{:.2f} ± {:.2f}".format(mean, std) if mean != 0 else ""
                for mean, std in zip(mean_row, std_row)
            ]
            for mean_row, std_row in zip(mean_mat, std_mat)
        ]
    )
    sns.heatmap(
        mean_mat,
        annot=annot,
        cmap="viridis",
        fmt="",
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    plt.title(
        f"Parameter Overlap \n for {lang} Across All Tasks",
        fontsize=15,
        fontweight="bold",
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"{save_path}/{lang}_jaccard_all")
    plt.close()


def jaccard_all(masks_list):
    num_models = len(masks_list[0])
    heatmap = np.zeros((num_models, num_models, len(masks_list)))

    for k, masks in enumerate(masks_list):
        for i in range(num_models):
            for j in range(i + 1, num_models):
                heatmap[i][j][k] = jaccard_all_params_from(masks[i](), masks[j]())
    return heatmap


def plot_jaccard_layers(layer_overlap_map, lang, save_path):
    for label, v in layer_overlap_map.items():
        plt.plot(
            v[0],
            np.mean(np.array(v[1]), axis=0),
            marker="o",
            label=label,
        )

    plt.title(
        f"Parameter Overlap Per Layer \n for {lang} Across All Tasks",
        fontsize=15,
        fontweight="bold",
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f"{save_path}/{lang}_jaccard_layers")
    plt.close()


def jaccard_layers(masks_list, datasets):
    x_labels = datasets
    y_labels = datasets

    def _label(i, j):
        return f"{x_labels[i]}-{y_labels[j]}"

    # if multiple masks are passed, we compute avg of layer overlaps.
    num_models = len(masks_list[0])
    # layer_overlap_map[label] = [x-axis, [y-axis..]] where x-axis is layers number and y-axis is the overlap ratio
    layer_overlap_map = {}
    for masks in masks_list:
        for i in range(num_models):
            for j in range(i + 1, num_models):
                layers = jaccard_layer_from(masks[i](), masks[j]())
                x = []
                y = []
                for l in layers.keys():
                    x.append(l)
                    # overlap / union
                    y.append(layers[l][0] / layers[l][1])
                label = _label(i, j)
                if label not in layer_overlap_map:
                    layer_overlap_map[label] = [x, [y]]
                    continue
                layer_overlap_map[label][1].append(y)

    return layer_overlap_map


def _mask(model_path, dataset):
    model, _, _ = build_model_tokenizer(model_path, dataset)
    mask = extract_mask(model, _EXCLUDED_LAYERS)
    return mask


def _process_inputs(models):
    seeds_to_models = {}
    for model_path, dataset, seed in models:
        p = (model_path, dataset)
        print(p)
        if seed not in seeds_to_models:
            seeds_to_models[seed] = [p]
            continue
        seeds_to_models[seed].append(p)
    
    # sort the dataset to aligned model between seed.
    for key, v in seeds_to_models.items():
        v.sort(key=lambda x: x[1])
        seeds_to_models[key] = v

    _check_alignment_beween_seeds(seeds_to_models)
    return seeds_to_models


def _dataset(model_dataset_list):
    print(model_dataset_list)
    return [dataset for _, dataset in model_dataset_list]

def _check_dup(datasets):
     if len(set(_dataset(datasets))) != len(datasets):
            raise Exception(f"duplicated dataset: {datasets}")

def _check_alignment_beween_seeds(seeds_to_models):
    items = list(seeds_to_models.items())
    _, prev_v = items[0]
    _check_dup(prev_v)
    for key, v in items[1:]:
        if _dataset(v) != _dataset(prev_v):
            raise Exception(
                f"model not aligned between seeds: prev {prev_v[1]} and curr {v[1]}"
            )
        _check_dup(v)
        prev_v = v


def main(args):
    print(args)
    masks_list = []
    seed_to_model = _process_inputs(args.model)
    for _, models_info in seed_to_model.items():
        masks = []
        for model_path, dataset in models_info:
            masks.append(lambda path=model_path, dataset=dataset: _mask(path, dataset))
        masks_list.append(masks)

    # just use any one of the dataset because they all have to be the same.
    x_labels = _dataset(list(seed_to_model.values())[0])
    heap_map = jaccard_all(masks_list)
    plot_jaccard_all(x_labels, heap_map, args.lang, args.saved_plots)
    layer_ratio_map = jaccard_layers(masks_list, x_labels)
    plot_jaccard_layers(layer_ratio_map, args.lang, args.saved_plots)


if __name__ == "__main__":
    parser = eval_parser()
    main(parser.parse_args())
