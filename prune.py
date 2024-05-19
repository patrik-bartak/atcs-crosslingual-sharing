# The current pruning config is based off of Prasanna's method (with some adjustments due to
# needing to find language specific subnetworks while also only using English for training)

import os
import json

import datasets
import torch
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


# We require a custom callback to evaluate at the right times
class AccuracyStoppingCallback(TrainerCallback):
    def __init__(
            self, trainer, original_acc, target_percent, savedir, interval
    ) -> None:
        super().__init__()

        self._trainer = trainer
        self.interval = interval
        self.stopping_acc = original_acc
        self.target_percent = target_percent

        # For saving
        self.acc_list = []
        self.spar_list = []
        self.model_savedir = savedir
        self.state_savedir = f"{savedir}/state.json"

        # Ensure savedir exists
        os.makedirs(savedir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        # Force the evaluation to happen at specific intervals based on the number of
        # batches per epoch (unfortunately has to be done due to limitations with neural-compressor)
        if ((state.global_step - 2) % self.interval == 0) and (
                (state.global_step - 2) > 0
        ):

            eval_metrics = self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval"
            )

            eval_accuracy = eval_metrics["eval_accuracy"]
            current_sparsity = ((state.global_step - 2) // self.interval) * 0.05
            self.acc_list.append(eval_accuracy)
            self.spar_list.append(current_sparsity)

            trainer_state = {
                "sparsity": current_sparsity,
                "accuracy": eval_accuracy,
                "spar_progression": self.spar_list,
                "acc_progression": self.acc_list,
                "configs": self._trainer.args.to_json_string(),
            }

            with open(self.state_savedir, "w") as outfile:
                json.dump(trainer_state, outfile, indent=2)

            if (
                    eval_accuracy < self.stopping_acc * self.target_percent
            ):  # To ensure the accuracy stays within the target range

                print(
                    f"\nAccuracy below {round(self.stopping_acc * self.target_percent, 4)}. Stopping Training...\n"
                )
                control.should_training_stop = True

            else:  # Else we save the second best checkpoint manually (not possible with default classes)

                self._trainer.model.save_pretrained(self.model_savedir)
                print(
                    f"\nCurrent Evaluation Accuracy: {round(eval_accuracy, 4)} | Step: {state.global_step}"
                )
                print(f"Model saved to {self.model_savedir}!\n")

            return control


def build_model_tokenizer_metric(model_name, tok_name, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        metric = None  # TODO: Implement

    elif dataset_name == SIB200:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = compute_acc

    elif dataset_name == XNLI:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = compute_acc  # TODO: Check if correct

    elif dataset_name == MQA:
        model = AutoModelForMultipleChoice.from_pretrained(
            model_name
        )  # I assume this one is what we need
        metric = None  # TODO: Implement

    else:
        raise Exception(f"Dataset {dataset_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return model, tokenizer, metric


def build_trainer_args(args):
    return TrainingArguments(
        output_dir=f"{args.model}-{args.dataset}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=args.lr,
        do_train=True,
        do_eval=False,
        bf16=False,
        max_steps=-1,
    )


def build_pruning_config(args, interval):
    return WeightPruningConfig(
        target_sparsity=0.5,  # So we can actually prune more (because the default is 0.5)
        pruning_type=args.type,
        start_step=1,
        end_step=int(args.epochs * interval) + 1,
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
    model, tokenizer, metric = build_model_tokenizer_metric(
        args.model, args.tokenizer, args.dataset
    )
    hf_dataset, lang_list, tokenize_fn = get_test_data(args.dataset)
    data_collator = get_data_collator(args.dataset, tokenizer)

    for lang in lang_list:

        print(f"Pruning for Language: {lang}")

        try:
            dataset = load_dataset(hf_dataset, lang)
        except Exception:
            dataset = datasets.load_from_disk(f"data/{lang}_xnli_val")

        val_dataset = dataset[
            "validation"
        ]  # May need to be adjusted for each dataset

        val_dataset = val_dataset.map(
            partial(tokenize_fn, tokenizer=tokenizer),
            batched=True,
            num_proc=4,
        ).filter(filter_context_length)

        # Need to create a label column for SIB200
        if hf_dataset == SIB200:
            # Map categories to labels
            val_dataset = val_dataset.map(map_categories_to_labels)
            val_dataset = val_dataset.remove_columns("category")

        print(f"Dset sizes (val): ({len(val_dataset)})")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device == "cpu":
            raise Exception("Should not train on CPU")

        # Need this part to align everything (the pruning happens every epoch (step))
        interval = ceil(len(val_dataset) / args.batch_size)
        pruning_config = build_pruning_config(args, interval)
        arguments = build_trainer_args(args)

        for seed in args.seed:
            model_c = deepcopy(model)  # Ensure the model is different every time
            output_dir = f"{args.savedir}/{args.type}-{lang}-{seed}"
            arguments.seed = seed
            trainer = INCTrainer(
                model=model_c,
                pruning_config=pruning_config,
                args=arguments,
                train_dataset=val_dataset,  # I assume we only ever use the dev dataset
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=metric,
            )

            # To get the accuracy on the test
            out = trainer.evaluate(val_dataset, metric_key_prefix="eval")
            orig_acc = out["eval_accuracy"]

            print(f"\nSeed: {seed}")
            print(f"Original Accuracy: {orig_acc}")

            trainer.add_callback(
                AccuracyStoppingCallback(
                    trainer, orig_acc, args.target_percent, output_dir, interval
                )
            )
            trainer.train()

            out_c = trainer.evaluate(val_dataset, metric_key_prefix="eval")
            curr_acc = out_c["eval_accuracy"]

            print(f"Current Accuracy: {curr_acc}")


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
