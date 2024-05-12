import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.dataset import *
from datasets import load_metric
from parsing import get_finetune_parser


# For getting the accuracy
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def build_model_tokenizer(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(
            hf_model_id, num_labels=7  # From 0 to 6
        )

    elif dataset_name == XNLI:
        return NotImplemented

    elif dataset_name == SIB200:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=7
        )  # Seven different categories

    elif dataset_name == MQA:
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        no_cuda=not args.cuda,
        bf16=False,
        max_steps=1 if args.test_run else args.max_steps,
    )


def main(args):
    print(args)
    model, tokenizer = build_model_tokenizer(args.model, args.dataset)
    train_dataset, val_dataset = build_dataset(args.dataset, tokenizer)
    data_collator = get_data_collator(args.dataset, tokenizer)
    trainer = Trainer(
        model=model,
        args=build_trainer_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.savedir)

    # Testing on languages
    hf_dataset, lang_list, tokenize_fn = get_test_data(args.dataset)
    test_model(trainer, tokenizer, hf_dataset, lang_list, tokenize_fn)


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
