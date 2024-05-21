import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from parsing import get_finetune_parser
from utils.dataset import *

import evaluate
import numpy as np


def build_model_tokenizer(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(
            hf_model_id, num_labels=7  # From 0 to 6
        )

    elif dataset_name == XNLI:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=3
        )  # 3 different categories

    elif dataset_name == SIB200:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=7
        )  # Seven different categories

    elif dataset_name == TOXI:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=2
        )  # 2 different categories
        metric = compute_acc

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer


def build_trainer_args(args):
    return TrainingArguments(
        output_dir=f"{args.model}-{args.dataset}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        no_cuda=not args.cuda,
        bf16=False,
        max_steps=(
            1 if args.test_run else (-1 if args.no_max_steps else args.max_steps)
        ),
        seed=args.seed[0],
        save_total_limit=3,
    )

def get_compute_metrics_fn():
    metric = evaluate.load('accuracy')

    def accuracy(eval_pred):
        preds, labs = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labs)

    return accuracy


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


def test_model(trainer, tokenizer, hf_dataset, lang_list, tokenize_fn):

    for lang in lang_list:
        if hf_dataset == TOXI:
            dataset = load_dataset(TOXI, ignore_verifications=True)
            dataset = dataset.filter(lambda example: example['lang'] == lang)
            dataset = dataset.rename_column('is_toxic', 'label')
            dataset = dataset.remove_columns("lang")
        else:
            dataset = load_dataset(hf_dataset, lang)

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

        test_data = tok_dataset["test"]
        out = trainer.evaluate(test_data, metric_key_prefix="test")
        print(f"Results for Language '{lang}':")
        print(out, "\n")



def main(args):
    print(args)
    model, tokenizer = build_model_tokenizer(args.model, args.dataset)

    train_dataset, val_dataset = build_dataset(args.dataset, tokenizer)
    print(f"Dset sizes (train/val): ({len(train_dataset)}/{len(val_dataset)})")
    data_collator = get_data_collator(args.dataset, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cpu":
        raise Exception("Should not train on CPU")

    print(train_dataset[:50]["label"])
    # val_dataset = val_dataset.select(range(10))

    trainer = Trainer(
        model=model,
        args=build_trainer_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not args.test_run else None,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_fn(),
    )
    if not args.no_do_train:
        trainer.train(resume_from_checkpoint=args.resume_path)
        trainer.save_model(args.savedir)

    # Testing on languages
    hf_dataset, lang_list, tokenize_fn = get_test_data(args.dataset)
    test_model(trainer, tokenizer, hf_dataset, lang_list, tokenize_fn, split="validation")


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())