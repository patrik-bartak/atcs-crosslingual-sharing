import os
import json
import numpy as np
import matplotlib.pyplot as plt

def get_lang_list(task):
    if task == 'SIB200':
        langs = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]

    elif task == 'XNLI':
        langs = ["cs", "hi", "id", "nl", "zh"]
    
    elif task == 'WIKIANN':
        langs = ["cs", "hi", "id", "nl", "zh"]
        
    elif task == 'TOXI':
        langs = ["hi", "zh-cn", "cs", "nl", "id"]

    else:
        return NotImplemented

    return langs

def parse_state_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["spar_progression"], data["acc_progression"]


#specify base_dir & task
base_dir = "pruned_models_states"

task = "XNLI"

base_dir = os.path.join(base_dir, task)

seeds = ["41", "42", "43"]

languages = get_lang_list(task)

all_sparsity_data = {}
all_accuracy_data = {}

for language in languages:
    sparsity_data = []
    accuracy_data = []
    for seed in seeds:
        file_path = os.path.join(base_dir, f"snip-{language}-{seed}", "state.json")
        if os.path.exists(file_path):
            sparsity, accuracy = parse_state_json(file_path)
            sparsity_data.append(sparsity)
            accuracy_data.append(accuracy)
            
    all_sparsity_data[language] = sparsity_data
    all_accuracy_data[language] = accuracy_data

for language in languages:
    mean_accuracy = np.mean(all_accuracy_data[language], axis=0)
    plt.figure()
    for i, seed in enumerate(seeds):
        plt.plot(all_sparsity_data[language][i], all_accuracy_data[language][i], marker='o', label=f"Seed {seed}")
    plt.plot(all_sparsity_data[language][0], mean_accuracy, linestyle='--', color='black', label="Mean Accuracy")
    plt.xlabel("Sparsity Progression")
    plt.ylabel("Accuracy Progression")
    plt.title(f"{task} : Accuracy Progression with Increase in Sparsity Levels ({language})")
    plt.grid(True)
    plt.legend()
    plot_dir = f'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/{task}_{language}_acc_prog.png')
    plt.close()