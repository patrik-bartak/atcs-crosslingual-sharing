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

    elif dataset_name == MARC:
        return NotImplemented

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
        evaluation_strategy="steps",
        eval_steps=1_000,
        save_strategy="steps",
        save_steps=1_000,
        load_best_model_at_end=True,
        learning_rate=args.lr,
        no_cuda=not args.cuda,
        use_cpu=not args.cuda,
        bf16=False,
        max_steps=1 if args.test_run else args.max_steps,
    )


def get_compute_metrics_fn():
    metric = evaluate.load('accuracy')

    def accuracy(eval_pred):
        preds, labs = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labs)

    return accuracy


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
    trainer = Trainer(
        model=model,
        args=build_trainer_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not args.test_run else None,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_fn()
    )
    trainer.train()


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
