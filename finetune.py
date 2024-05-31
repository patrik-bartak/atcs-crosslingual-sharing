import torch
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
from utils.dataset import *

import evaluate
import numpy as np


def build_model_tokenizer_metric(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model=None
        metric = None

    elif dataset_name == XNLI:

        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=3
        )  # 3 different categories

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

    elif dataset_name == TOXI:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=2
        )  # 2 different categories
        metric = compute_acc

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer, metric

def build_model_wikiann(hf_model_id, dataset_name, raw_dataset):
  if dataset_name == WIKIANN:
      ner_feature = raw_dataset["train"].features["ner_tags"]
      label_names = ner_feature.feature.names
      labels = raw_dataset["train"][0]["ner_tags"]
      labels = [label_names[i] for i in labels]
      id2label = {i: label for i, label in enumerate(label_names)}
      label2id = {v: k for k, v in id2label.items()}
      model = AutoModelForTokenClassification.from_pretrained(
      hf_model_id, num_labels=7, id2label=id2label,label2id=label2id)
  else:
        raise Exception(f"Dataset {dataset_name} not supported")
  
  return model

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
        max_steps=(
            1 if args.test_run else (-1 if args.no_max_steps else args.max_steps)
        ),
        seed=args.seed[0],
        save_total_limit=3,
    )


def get_compute_metrics_fn():
    metric = evaluate.load("accuracy")

    def accuracy(eval_pred):
        preds, labs = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labs)

    return accuracy


def main(args):
    print(args)
    model, tokenizer, metric = build_model_tokenizer_metric(args.model, args.dataset)
    
    if args.dataset =='wikiann':
        raw_dataset, train_dataset, val_dataset= build_dataset_wikiann(args.dataset, tokenizer)
        print(f"Dset sizes (train/val): ({len(train_dataset)}/{len(val_dataset)})")
        data_collator = get_data_collator(args.dataset, tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device == "cpu":
            raise Exception("Should not train on CPU")
        model = build_model_wikiann(args.model, args.dataset, raw_dataset)
        
        trainer = Trainer(
            model=model,
            args=build_trainer_args(args),
            train_dataset=train_dataset,
            eval_dataset=val_dataset if not args.test_run else None,
            data_collator=data_collator,
            compute_metrics=get_ner_metrics,
        )
    else:
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
