from functools import partial

from datasets import load_dataset
from transformers import (DataCollatorForTokenClassification,
                          DataCollatorWithPadding)

XML_R = "FacebookAI/xlm-roberta-base"

XNLI = "xnli"
SIB200 = "Davlan/sib200"
WIKIANN = "wikiann"
MQA = "clips/mqa"


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(input, tokenizer):
    tokenized_inputs = tokenizer(
        input["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = input["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


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
    return tokenize_and_align_labels(rows, tokenizer)


def tokenize_mqa(rows, tokenizer):
    return tokenizer(NotImplemented, return_special_tokens_mask=True)


def build_dataset(hf_dataset, tokenizer):
    if hf_dataset == XNLI:
        dataset = load_dataset(XNLI, "en")
        tokenize_fn = tokenize_xnli
    elif hf_dataset == SIB200:
        # TODO: load the correct data language subset
        dataset = load_dataset(SIB200, "eng_Latn")
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
    # Some datasets may need us to manually call .train_test_split() to
    # get the splits if they are not predefined
    train_dataset = tok_dataset["train"]
    val_dataset = tok_dataset["validation"]

    return train_dataset, val_dataset
