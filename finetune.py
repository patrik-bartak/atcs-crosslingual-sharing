import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)

from utils.dataset import *
from utils.constants import *
from parsing import get_finetune_parser


def build_model_tokenizer_metric(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(
            hf_model_id, num_labels=7  # From 0 to 6
        )
        metric = None  # TODO: Implement

    elif dataset_name == XNLI:
        model = None  # TODO: Implement
        metric = compute_acc
        return NotImplemented

    elif dataset_name == SIB200:

        # For editing the idx2label and label2idx settings
        config = AutoConfig.from_pretrained(hf_model_id)
        config.id2label = sib_idx2cat
        config.label2id = sib_cat2idx
        config.num_labels = 7
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, config=config
        )  # Seven different categories
        metric = compute_acc

    elif dataset_name == MQA:
        return NotImplemented

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer, metric


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


def main(args):
    print(args)
    model, tokenizer, metric = build_model_tokenizer_metric(args.model, args.dataset)
    train_dataset, val_dataset = build_dataset(args.dataset, tokenizer)
    data_collator = get_data_collator(args.dataset, tokenizer)
    trainer = Trainer(
        model=model,
        args=build_trainer_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metric,
    )

    trainer.train()
    trainer.save_model(args.savedir)

    # Testing on languages
    hf_dataset, lang_list, tokenize_fn = get_test_data(args.dataset)
    test_model(trainer, tokenizer, hf_dataset, lang_list, tokenize_fn)


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
