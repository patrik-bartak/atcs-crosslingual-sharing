from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from dataset import WIKIANN, build_dataset, get_data_collator
from parsing import get_finetune_parser


def build_model_tokenizer(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(hf_model_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
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
        eval_dataset=val_dataset if not args.test_run else None,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
