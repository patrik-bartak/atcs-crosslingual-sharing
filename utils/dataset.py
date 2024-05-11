from functools import partial
from utils.constants import *
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding


def get_data_collator(hf_dataset, tokenizer):
    return (
        DataCollatorForTokenClassification(tokenizer=tokenizer)
        if hf_dataset == WIKIANN
        else DataCollatorWithPadding(tokenizer=tokenizer)
    )


def tokenize_xnli(rows, tokenizer):
    return tokenizer(
        rows["premise"], rows["hypothesis"], return_special_tokens_mask=True
    )


def tokenize_sib200(rows, tokenizer):
    return tokenizer(NotImplemented, return_special_tokens_mask=True)


def tokenize_wikiann(rows, tokenizer):
    return tokenizer(NotImplemented, return_special_tokens_mask=True)


def tokenize_mqa(rows, tokenizer):
    return tokenizer(NotImplemented, return_special_tokens_mask=True)


def build_dataset(hf_dataset, tokenizer):
    if hf_dataset == XNLI:
        dataset = load_dataset(XNLI, "en")
        tokenize_fn = tokenize_xnli

    elif hf_dataset == SIB200:
        # TODO: load the correct data language subset
        dataset = load_dataset(SIB200)
        tokenize_fn = tokenize_sib200

    elif hf_dataset == WIKIANN:
        dataset = load_dataset(WIKIANN, "en")
        tokenize_fn = tokenize_wikiann

    elif hf_dataset == MQA:
        # TODO: load the correct data language subset
        dataset = load_dataset(MQA)
        tokenize_fn = tokenize_mqa

    else:
        raise Exception(f"Dataset {hf_dataset} not supported")

    tok_dataset = dataset.map(
        partial(tokenize_fn, tokenizer=tokenizer),
        batched=True,
        num_proc=4,
    )
    # Some datasets may need us to manually call .train_test_split() to get the splits
    train_dataset = tok_dataset["train"]
    val_dataset = tok_dataset["test"]

    return train_dataset, val_dataset
