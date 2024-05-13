# For consistency

# The model
XML_R = "FacebookAI/xlm-roberta-base"

# The HuggingFace datasets
XNLI = "xnli"
SIB200 = "Davlan/sib200"
WIKIANN = "wikiann"
MQA = "clips/mqa"

# For SIB200 (converting category to label)
sib_cat2idx = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}

sib_idx2cat = {v: k for k, v in sib_cat2idx.items()}

# For getting the correct test datasets
# SIB200
lang_sib = ["ces_Latn", "hin_Deva", "ind_Latn", "nld_Latn", "zho_Hans"]

# WikiAnn
lang_wik = ["cs", "hi", "id", "nl", "zh"]

# XNLI (need three other datasets)
lang_xnli = ["hi", "zh"]
