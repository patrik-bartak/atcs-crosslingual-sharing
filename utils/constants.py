# For consistency

# The model
XML_R = "FacebookAI/xlm-roberta-base"

# The HuggingFace datasets
XNLI = "xnli"
SIB200 = "Davlan/sib200"
WIKIANN = "wikiann"
MQA = "clips/mqa"

# Directory of fine-tuned models:
FT_WIKI = "/ft_models/WikiAnn"
FT_XNLI = "/ft_models/XNLI"
FT_MQNA = "/ft_models/MQA"
FT_TOPP = "/ft_models/TOP"

# Directory of pruned models:
PR_WIKI = "/pruned_models/WikiAnn"
PR_XNLI = "/pruned_models/XNLI"
PR_MQNA = "/pruned_models/MQA"
PR_TOPP = "/pruned_models/WikTOPAnn"
