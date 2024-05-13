import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
import evaluate
from utils.dataset import *
from utils.constants import *
from datasets import load_metric
from parsing import get_finetune_parser


# For getting the accuracy
metric = load_metric("accuracy", trust_remote_code=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Assuming your model outputs logits
    predictions = np.argmax(predictions, axis=1)
    # If labels are one-hot encoded, convert them to categorical labels
    labels = np.argmax(labels, axis=1)
    return {"accuracy": (predictions == labels).mean()}


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


def build_model_tokenizer(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = None
    
    elif dataset_name == XNLI:
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

    elif dataset_name == MQA:
        return NotImplemented

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer


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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        no_cuda=not args.cuda,
        bf16=False,
        max_steps=1 if args.test_run else args.max_steps,
        seed=args.seed,
    )


def main(args):
    print(args)
    model, tokenizer = build_model_tokenizer(args.model, args.dataset)
    
    if args.dataset =='wikiann':
      raw_dataset, train_dataset, val_dataset= build_dataset_wikiann(args.dataset, tokenizer)
      data_collator = get_data_collator(args.dataset, tokenizer)
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
      data_collator = get_data_collator(args.dataset, tokenizer)
      trainer = Trainer(
          model=model,
          args=build_trainer_args(args),
          train_dataset=train_dataset,
          eval_dataset=val_dataset if not args.test_run else None,
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
