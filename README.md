# SimpleFLUX
This repository contains a simple and flexible PyTorch implementation of [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) based on diffusers.
The main purpose is to make it easier for generative model researchers to do DIY design and fine-tuning based on the powerful FLUX.1 model.

<div align="center">
   <img src="./SimpleFLUX.png">
</div>

## Limitations
- Please note that as FLUX.1 is a **12B** model, at bf16 precision, the inference of this model requires approximately **45G** of GPU memory. (can be reduced to 24GB if you use `inference_original.py` as it sequentially offload used modules to cpu.)
- Training only all `transformer_blocks` requires about 60G of GPU memory, and we need **deepspeed** (with cpu offload) to train all parameters of this large model.
- Better GPU memory optimization is expected to be investigated.

## Prepartion
- You should download the diffusers version checkpoints of FLUX.1-dev, from [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), including scheduler, text_encoder(s), tokenizer(s), transformer, and vae. Then put it in the ckpt folder.
- You also can download the model in python script (Note: You should login with your huggingface token first.):

```
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir="./ckpt")
```

- If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.

```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir ckpt --local-dir-use-symlinks False
```

## Requirements
- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.3.1](https://pytorch.org/)
- triton == 2.3.1
- diffusers == 0.32.1
- accelerate == 1.2.1
- transformers == 4.46.3
- xformers == 0.0.27
- bitsandbytes == 0.45.0
- deepspeed == 0.16.2

A suitable [conda](https://conda.io/) environment named `simpleflux` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate simpleflux
```

## Dataset Preparation
- You need write a DataLoader suitable for your own Dataset, because we just provide a simple example to test the code.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."
```

## Acknowledgements
Many thanks to the checkpoint from [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/) and code bases from [diffusers](https://github.com/huggingface/diffusers/), [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM/) and [SimpleSDM-3](https://github.com/haoningwu3639/SimpleSDM-3).