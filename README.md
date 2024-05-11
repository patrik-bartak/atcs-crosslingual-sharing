# **Crosslingual Sharing**
## **Identifying Subnetworks for languages**

The code is base on this [Hugging Face Template](https://github.com/MorenoLaQuatra/transformers-tasks-templates/).

## **Installation**

Download this repo into your local machine:

```bash
$ cd <your workspace>
$ git clone https://github.com/patrik-bartak/atcs-crosslingual-sharing.git
$ cd atcs-crosslingual-sharing
```

Setup python environment using conda or mamba. If you haven't installed mamba on your machine, follow this [guide](https://www.usna.edu/Users/cs/fknoll/SD211/mamba.html).

```bash
$ conda env create -f gpu_env.yml
```

```bash
$ mamba env create -f gpu_env.yml
```

## **Model Finetuning**

To finetune a model, run the following script:
```bash
$ python finetune.py --dataset <insert_dataset_name_here> \ ...
```
The list of arguments that can be specified are the following:



## **Model Pruning**

To prune a model, run the following script:
```bash
$ python prune.py --dataset <insert_dataset_name_here> \ ...
```
The list of arguments that can be specified are the same as with fine-tuning, with some additional arguments:

