from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from args import parse_arguments
from data_ultil import process_dataset


def _build_model_tokenizer(hf_model_id):
    model = AutoModelForMaskedLM.from_pretrained(hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer


def _build_trainer_args(args):
    return TrainingArguments(
        output_dir=f"{args.model}-{args.dataset}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        no_cuda=not (args.cuda),
        bf16=True,
    )


def main():
    model, tokenizer = _build_model_tokenizer(args.model)
    train_dataset, val_dataset = process_dataset(args.dataset, args.split, tokenizer)
    trainer = Trainer(
        model=model,
        args=_build_trainer_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=True, mlm_probability=args.mlm_prob
        ),
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_arguments()
    main()
