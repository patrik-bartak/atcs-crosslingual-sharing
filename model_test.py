from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
#
# # prepare input
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
#
# # forward pass
# output = model(**encoded_input)
#
#
# from transformers import AutoModelForSequenceClassification
#
# model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
#
# # prepare input
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
#
# # forward pass
# output2 = model(**encoded_input)

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=7
)


# # prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
#
# # forward pass
output3 = model(**encoded_input)
