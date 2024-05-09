import argparse

from dataset import XML_R


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
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to be used for training.",
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
    )

    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Test run or not",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )

    return parser
