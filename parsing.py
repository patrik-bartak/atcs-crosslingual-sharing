import argparse
from utils.constants import *


def get_finetune_parser():
    """
    Get arg parser for the finetuning script.
    :return: The parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=XML_R,
        help="The identifier for the model to be used. It can be an identifier from the transformers library or a "
        "path to a local model.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=XML_R,
        help="The identifier for the tokenizer to be used. Must be an existing HuggingFace tokenizer.",
    )

    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="The path to save the best model to.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to be used for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        nargs="+",
        help="The seed to use for training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to be used for training.",
    )

    parser.add_argument(
        "--checkpts",
        type=str,
        default="checkpoints",
        help="The directory where the checkpoints will be saved.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="The directory where the logs will be saved.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="The learning rate to be used for training.",
    )

    parser.add_argument(
        "--cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable cuda computation",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="hugging face dataset name",
        choices={XNLI, SIB200, WIKIANN, TOXI},
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run or not",
    )

    parser.add_argument(
        "--no-max-steps",
        action="store_true",
        help="Whether we use max-steps at all",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of training steps",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="magnitude",
        help="Type of pruning to use",
        choices={"magnitude", "gradient", "snip"},
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Percentage of neurons to mask per step",
    )

    parser.add_argument(
        "--target_percent",
        type=float,
        default=0.9,
        help="The percentage of the original performance to keep",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="1x1",
        help="The pruning pattern to use",
    )

    return parser

def eval_parser():
    """
    Get arg parser for the finetuning script.
    :return: The parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs=3,
        type=str,
        action='append',
        metavar=('local_path','dataset', "seed"),
        help="We pass in the model path along with metadata about trained dataset and seed",
    )

    parser.add_argument(
        "--lang",
        type=str,
        help="Language that model is pruned on.",
    )

    parser.add_argument(
        "--saved-plots",
        type=str,
        help="path for storing plots.",
    )
    return parser