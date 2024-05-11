# The current pruning config is based off of Prasanna's method

from copy import deepcopy
from utils.constants import *
from optimum.intel import INCTrainer
from parsing import get_finetune_parser
from neural_compressor import WeightPruningConfig
from utils.dataset import build_dataset, get_data_collator
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)


# Need to adjust this more
class AccuracyStoppingCallback(TrainerCallback):
    def __init__(self, trainer, target_percent) -> None:
        super().__init__()
        self._trainer = trainer
        self.stopping_acc = None
        self.target_percent = target_percent

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            eval_metrics = self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            eval_accuracy = eval_metrics["eval_accuracy"]

            if self.stopping_acc == None:  # For the base accuracy
                self.stopping_acc = eval_accuracy

            elif (
                eval_accuracy < self.stopping_acc * self.target_percent
            ):  # To ensure the accuracy stays within the target range
                control.should_training_stop = True

            return control_copy


# Update this to load from "ft_models" (and probably only depend on the task dataset)
def build_model_tokenizer(hf_model_id, dataset_name):
    if dataset_name == WIKIANN:
        model = AutoModelForTokenClassification.from_pretrained(hf_model_id)
        save_dir = PR_WIKI
    else:
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        save_dir = PR_XNLI

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    return model, tokenizer, save_dir


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
        do_train=True,
        do_eval=True,
        bf16=False,
        max_steps=1 if args.test_run else args.max_steps,
    )


# Might need to specify excluded layers (i.e., embeddings)
def build_pruning_config(args):
    return WeightPruningConfig(
        target_sparsity=0.5,  # So we can actually prune more (default is 0.9)
        pruning_type=args.type,
        start_step=1,  # Seems to me that we can prune per epoch using the training argument
        end_step=1,
        min_sparsity_ratio_per_op=args.sparsity,
        max_sparsity_ratio_per_op=args.sparsity,
        pruning_scope="global",
    )


def main(args):
    print(args)
    model, tokenizer, save_dir = build_model_tokenizer(args.model, args.dataset)
    pruning_config = build_pruning_config(args)
    _, val_dataset = build_dataset(args.dataset, tokenizer)
    data_collator = get_data_collator(args.dataset, tokenizer)
    trainer = INCTrainer(
        model=model,
        pruning_config=pruning_config,
        args=build_trainer_args(args),
        train_dataset=val_dataset,  # I assume we only ever use the dev dataset
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(AccuracyStoppingCallback(trainer, args.target_percent))
    trainer.train()
    trainer.save_model(save_dir)


if __name__ == "__main__":
    parser = get_finetune_parser()
    main(parser.parse_args())
