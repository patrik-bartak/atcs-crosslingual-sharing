# atcs-crosslingual-sharing

The code is base on this [Hugging Face Template](https://github.com/MorenoLaQuatra/transformers-tasks-templates/).

# Installation

Download this repo into your local machine:

```bash
$ cd <your workspace>
$ git clone https://github.com/patrik-bartak/atcs-crosslingual-sharing.git
$ cd atcs-crosslingual-sharing
```

Setup python environment using conda or mamba. If you haven't installed mamba on your machine, follow this [guide](https://www.usna.edu/Users/cs/fknoll/SD211/mamba.html).

```bash
$ mamba env create -f gpu_env.yml
```

## Finetune Model

See 'atcs-crosslingual-sharing/fill-mask/README.MD' as an example how to finetune fill-mask task.
We will need to create `atcs-crosslingual-sharing/<task>/` for different LLM tasks. 