from utils import SIB200, XNLI


def get_lang_list(task):
    if task == SIB200:
        langs = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]

    elif task == XNLI:
        langs = ["cs", "hi", "id", "nl", "zh", "en"]

    else:
        langs = ["cs", "hi", "id", "nl", "zh"]

    return langs

