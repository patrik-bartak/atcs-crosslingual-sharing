# The current pruning config is based off of Prasanna's method (with some adjustments due to
# needing to find language specific subnetworks while also only using English for training)

import torch
import numpy as np
from math import ceil
from copy import deepcopy
from utils.dataset import *
from utils.constants import *
from datasets import load_metric
from optimum.intel import INCTrainer
from parsing import get_finetune_parser
from neural_compressor import WeightPruningConfig
from utils.dataset import get_data_collator
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)

# For getting the accuracy
metric = load_metric("accuracy", trust_remote_code=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Need to adjust this more
class AccuracyStoppingCallback(TrainerCallback):
    def __init__(self, trainer, original_acc, target_percent, savedir) -> None:
        super().__init__()

        self._trainer = trainer
        self.stopping_acc = original_acc
        self.target_percent = target_percent

        # For saving
        self.model_savedir = f"{savedir}/model.pt"
        self.state_savedir = f"{savedir}/state.pt"

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            eval_metrics = self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval"
            )

            eval_accuracy = eval_metrics["eval_accuracy"]

            if (
                eval_accuracy < self.stopping_acc * self.target_percent
            ):  # To ensure the accuracy stays within the target range

                print(
                    f"Accuracy below {self.stopping_acc * self.target_percent}. Stopping Training..."
                )
                control.should_training_stop = True

            else:  # Else we save the second best checkpoint manually (not possible with default classes)

                trainer_state = {
                    "step": self._trainer.state.global_step,
                    "configs": self._trainer.args.to_json_string(),
                }

                torch.save(self._trainer.model, self.model_savedir)
                torch.save(trainer_state, self.state_savedir)

            return control


def build_model_tokenizer(model_name, tok_name, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(model_name)

    elif dataset_name == SIB200:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    elif dataset_name == XNLI:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    elif dataset_name == MQA:
        model = AutoModelForMultipleChoice.from_pretrained(
            model_name
        )  # I assume this one is what we need

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return model, tokenizer


def build_trainer_args(args):
    return TrainingArguments(
        output_dir=f"{args.model}-{args.dataset}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        no_cuda=not args.cuda,
        do_train=True,
        do_eval=True,
        bf16=False,
        max_steps=-1,
        save_total_limit=2,
        save_only_model=True,
    )


# Might need to specify excluded layers (i.e., embeddings)
def build_pruning_config(args):
    return WeightPruningConfig(
        target_sparsity=0.5,  # So we can actually prune more (because the default is 0.5)
        pruning_type=args.type,
        start_step=1,
        end_step=51,
        pruning_scope="global",
        pruning_op_types=["Conv", "Linear", "Attention"],
        excluded_op_names=["roberta.embeddings"],  # Do not mask the embeddings
        pattern=args.pattern,
        pruning_frequency=1,
        max_sparsity_ratio_per_op=0.999,  # To enable a component being 'completely' sparse
        sparsity_decay_type="linear",
    )


def main(args):
    print(args)
    model, tokenizer = build_model_tokenizer(args.model, args.tokenizer, args.dataset)
    pruning_config = build_pruning_config(args)
    hf_dataset, lang_list, tokenize_fn = get_test_data(args.dataset)
    data_collator = get_data_collator(args.dataset, tokenizer)

    for lang in lang_list:

        print(f"Pruning for Language: {lang}")
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

        val_dataset = tok_dataset[
            "validation"
        ]  # May need to be adjusted for each dataset

        # Need this part to align everything (the pruning happens every epoch (step))
        step_base = ceil(len(val_dataset) / args.batch_size)
        pruning_config.end_step = int(step_base * args.epochs) + 1
        print(pruning_config.end_step)

        arguments = build_trainer_args(args)

        for seed in args.seed:

            model_c = deepcopy(model)  # Ensure the model is different every time
            arguments.output_dir = f"{args.model}-{lang}-{seed}"
            arguments.seed = seed
            trainer = INCTrainer(
                model=model_c,
                pruning_config=pruning_config,
                args=arguments,
                train_dataset=val_dataset,  # I assume we only ever use the dev dataset
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            # To get the accuracy on the test
            out = trainer.evaluate(val_dataset, metric_key_prefix="eval")
            orig_acc = out["eval_accuracy"]

            print(f"\nSeed: {seed}")
            print(f"Original Accuracy: {orig_acc}")

            trainer.add_callback(
                AccuracyStoppingCallback(
                    trainer, orig_acc, args.target_percent, arguments.output_dir
                )
            )
            trainer.train()

            save_dir = f"{args.savedir}/{seed}/{lang}"

            out_c = trainer.evaluate(val_dataset, metric_key_prefix="eval")
            curr_acc = out_c["eval_accuracy"]

            print(f"Current Accuracy: {curr_acc}")

            trainer.save_model(save_dir)


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
