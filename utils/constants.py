# For consistency

# The model
XML_R = "FacebookAI/xlm-roberta-base"

# The HuggingFace datasets
XNLI = "xnli"
SIB200 = "Davlan/sib200"
WIKIANN = "wikiann"
MARC = "marc"

# Directory of pruned models:
PR_WIKI = "/pruned_models/WikiAnn"
PR_XNLI = "/pruned_models/XNLI"
PR_MQNA = "/pruned_models/MARC"
PR_TOPP = "/pruned_models/SIB200"

# For SIB200 (converting category to label)
cat2idx = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}
