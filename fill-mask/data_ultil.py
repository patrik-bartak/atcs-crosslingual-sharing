from datasets import load_dataset

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def process_dataset(hg_dataset, split, tokenizer):
    
    train_dataset = None
    val_dataset = None

    def tokenize(examples):
        return tokenizer(examples["output"])

    if hg_dataset == "DavideTHU/chinese_news_dataset":
        datasets = load_dataset(hg_dataset, split=split)
        datasets = datasets.train_test_split(test_size=0.2)

        tokenized_datasets = datasets.map(
            tokenize,
            batched=True,
            num_proc=4,
            remove_columns=["output", "url", "instruction"],
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=4,
            fn_kwargs={"block_size": tokenizer.model_max_length},
        )
        train_dataset = lm_datasets['train']
        val_dataset = lm_datasets['test']

    # Add new datasets here.
    elif hg_dataset == "Davlan/sib200":
        datasets = load_dataset(hg_dataset, split=split)
        datasets = datasets.train_test_split(test_size=0.2)

        tokenized_datasets = datasets.map(
            tokenize,
            batched=True,
            num_proc=4,
        )

        train_dataset = lm_datasets['train']
        val_dataset = lm_datasets['test']

    else:
        raise Exception(f"Dataset {hg_dataset} not supported")

    return train_dataset, val_dataset
