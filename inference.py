import os
import torch
import random
import argparse
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from model.pipeline import FluxPipeline
from model.autoencoders import AutoencoderKL
from model.transformer import FluxTransformer2DModel

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/FLUX-1.dev', type=str)
    parser.add_argument('--prompt', default="A cat holding a sign that says hello world.", type=str)    
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    mixed_precision: Optional[str] = "bf16"   # "fp16", "no"
):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        
    accelerator = Accelerator(mixed_precision=mixed_precision)

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder_2 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")
    
    pipeline = FluxPipeline(
        vae = vae,
        transformer = transformer,
        scheduler = scheduler,
        text_encoder = text_encoder, 
        tokenizer = tokenizer,
        text_encoder_2 = text_encoder_2,
        tokenizer_2 = tokenizer_2,
    )
        
    transformer, pipeline = accelerator.prepare(transformer, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleFLUX")

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    transformer.eval()
    
    # sample_seed = random.randint(0, 100000)
    # generator = torch.Generator(device=accelerator.device)
    # generator.manual_seed(sample_seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(0)
    pipeline.enable_model_cpu_offload()
    
    output = pipeline(
        prompt = prompt,
        height = 1024,
        width = 1024,
        max_sequence_length=512,
        generator = generator,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
    )

    output_image = output.images[0] # PIL Image here
    output_image.save(os.path.join(logdir, f"{prompt}.png"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    mixed_precision = "bf16" # "fp16", "no"
    test(pretrained_model_path, logdir, prompt, num_inference_steps, guidance_scale, mixed_precision)

# CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."