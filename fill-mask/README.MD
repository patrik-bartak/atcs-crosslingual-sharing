# Fill Mask Task

This contains code to finetune model for Fill Mask Task. Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks. These models are useful when we want to get a statistical understanding of the language in which the model is trained in.

More info on [fill-mask](https://huggingface.co/tasks/fill-mask)

## Quick Start

Finetune 'xlm-roberta-base' model using `DavideTHU/chinese_news_dataset` dataset.

```
$ cd <your-workspace>/atcs-crosslingual-sharing
$ python atcs-crosslingual-sharing/fill-mask/train.py --model='FacebookAI/xlm-roberta-base' --split='train' --dataset='DavideTHU/chinese_news_dataset' --batch-size=1 
```

