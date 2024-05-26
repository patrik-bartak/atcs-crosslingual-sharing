import numpy as np
import evaluate
from functools import partial
from utils.constants import *
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding

# For getting the metrics
metric_acc = load_metric("accuracy", trust_remote_code=True)


def compute_acc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

def get_ner_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    d= compute_ner_metrics(labels, predictions)
    return d

def compute_ner_metrics(labels, predictions):
    # Remove ignored index (special tokens) and convert to labels
    metric = evaluate.load("seqeval")
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# Function to map categories to labels
def map_categories_to_labels(example):
    example["label"] = sib_cat2idx[example["category"]]
    return example


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


# General functions
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
    return tokenizer(rows["text"], return_special_tokens_mask=True)


def tokenize_wikiann(rows, tokenizer):
    return tokenize_and_align_labels(rows, tokenizer)


def tokenize_toxi(rows, tokenizer):
    return tokenizer(rows["text"], truncation=True, return_special_tokens_mask=True)


# For getting the test datasets (returns a list)
def get_test_data(hf_dataset):
    if hf_dataset == XNLI:
        return hf_dataset, lang_xnli, tokenize_xnli

    elif hf_dataset == SIB200:
        return hf_dataset, lang_sib, tokenize_sib200

    elif hf_dataset == WIKIANN:
        return hf_dataset, lang_wik, tokenize_wikiann
    
    elif hf_dataset == TOXI:
        return hf_dataset, lang_toxi, tokenize_toxi

    else:
        raise Exception(f"Value {hf_dataset} not supported")


# For evaluating per language
def test_model(trainer, tokenizer, hf_dataset, lang_list, tokenize_fn):

    for lang in lang_list:
        if hf_dataset == TOXI:
            dataset = load_dataset(hf_dataset, ignore_verifications=True)
            dataset = dataset.filter(lambda example: example['lang'] == lang)
            dataset = dataset.rename_column('is_toxic', 'label')
            dataset = dataset.remove_columns("lang")
        else:
            dataset = load_dataset(hf_dataset, lang)

        if hf_dataset == WIKIANN:
            tok_dataset = dataset.map(
                partial(tokenize_fn, tokenizer=tokenizer),
                batched=True,
                remove_columns=dataset["test"].column_names,
                num_proc=4)
        else:
            tok_dataset = dataset.map(
                partial(tokenize_fn, tokenizer=tokenizer),
                batched=True,
                num_proc=4,
            )

        # Need to create a label column for SIB200
        if hf_dataset == SIB200:
            # Map categories to labels
            tok_dataset = tok_dataset.map(map_categories_to_labels)
            tok_dataset = tok_dataset.remove_columns("category")
        if hf_dataset == TOXI:
            test_data = tok_dataset["train"]
        else:
            test_data = tok_dataset["test"]
        out = trainer.evaluate(test_data, metric_key_prefix="test")
        print(f"Results for Language '{lang}':")
        print(out, "\n")

def build_dataset_wikiann(hf_dataset, tokenizer):
  
    if hf_dataset == WIKIANN:
        dataset = load_dataset(WIKIANN, "en")
        tokenize_fn = tokenize_wikiann
    else:
        raise Exception(f"Dataset {hf_dataset} not supported")

    tok_dataset = dataset.map(
      partial(tokenize_fn, tokenizer=tokenizer),
      batched=True,
      remove_columns=dataset["train"].column_names,
      remove_columns=dataset["validation"].column_names,
      num_proc=4)
    # Some datasets may need us to manually call .train_test_split() to get the splits
    train_dataset = tok_dataset["train"]
    val_dataset = tok_dataset["validation"]

    return dataset, train_dataset, val_dataset


def build_dataset(hf_dataset, tokenizer):
    if hf_dataset == XNLI:
        dataset = load_dataset(XNLI, "en")
        tokenize_fn = tokenize_xnli

    elif hf_dataset == SIB200:
        dataset = load_dataset(SIB200, "eng_Latn")
        dataset = dataset.remove_columns("index_id")
        tokenize_fn = tokenize_sib200

    elif hf_dataset == TOXI:
        dataset = load_dataset(TOXI, ignore_verifications=True)
        dataset = dataset.filter(lambda example: example['lang'] == 'en')
        dataset = dataset.rename_column('is_toxic', 'label')
        dataset = dataset.remove_columns('lang')
        tokenize_fn = tokenize_toxi

    else:
        raise Exception(f"Dataset {hf_dataset} not supported")

    tok_dataset = dataset.map(
        partial(tokenize_fn, tokenizer=tokenizer),
        batched=True,
        num_proc=4,
    )

    # Need to create a label column for SIB200
    if hf_dataset == SIB200:
        # Map categories to labels
        tok_dataset = tok_dataset.map(map_categories_to_labels)
        tok_dataset = tok_dataset.remove_columns("category")

    # Some datasets may need us to manually call .train_test_split() to get the splits
    train_dataset = tok_dataset["train"]
    val_dataset = tok_dataset["validation"]

    return train_dataset, val_dataset
