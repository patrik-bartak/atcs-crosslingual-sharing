# This script is used to translate instances for languages that are not in XNLI
# First run "pip install argostranslate"

import argostranslate.package
import argostranslate.translate
from datasets import load_dataset
from constants import XNLI


def translate_xnli(to_lang, from_lang="en"):
    dataset = load_dataset(XNLI, from_lang)
    # Only translate the validation set
    del dataset["train"]
    del dataset["test"]

    # Save the original data for conversion
    dataset.save_to_disk("data/en_xnli_test_val")

    # Get the translation packages
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_lang and x.to_code == to_lang,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    def translate_fn(row):

        # print(f"Before: {row}")

        row["premise"] = argostranslate.translate.translate(
            row["premise"], from_lang, to_lang
        )

        row["hypothesis"] = argostranslate.translate.translate(
            row["hypothesis"], from_lang, to_lang
        )

        return row

        # print(f"After: {row}")

    dataset = dataset.map(translate_fn)

    return dataset


# Specify the languages here
langs_to_translate = ["nl", "id"]

for lang in langs_to_translate:

    print(f"Translating {lang}")
    data = translate_xnli(lang)
    data.save_to_disk(f"data/{lang}_xnli_test_val")
