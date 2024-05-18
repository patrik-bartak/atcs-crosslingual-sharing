import argostranslate.package
import argostranslate.translate
from datasets import load_dataset

from utils import XNLI


def translate_xnli(to_lang, from_lang="en"):
    dataset = load_dataset(XNLI, from_lang)
    # Only translate test and validation sets
    del dataset["train"]
    dataset.save_to_disk("data/en_xnli_test_val")

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
        row["premise"] = argostranslate.translate.translate(
            row["premise"], from_lang, to_lang
        )
        row["hypothesis"] = argostranslate.translate.translate(
            row["hypothesis"], from_lang, to_lang
        )

    return dataset.map(translate_fn)


langs_to_translate = ["cs", "id"]

for lang in langs_to_translate:
    print(f"Translating {lang}")
    translate_xnli(lang).save_to_disk(f"data/{lang}_xnli_test_val")
