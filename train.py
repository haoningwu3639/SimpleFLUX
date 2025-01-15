import os
import cv2
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Optional, Dict
from torch.cuda.amp import autocast
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, DeepSpeedPlugin # Use Deepspeed to reduce memory usage
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from dataset import SimpleDataset
from utils.util import get_time_string, get_function_args
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from model.pipeline import FluxPipeline
from model.autoencoders import AutoencoderKL
from model.transformer import FluxTransformer2DModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

class SampleLogger:
    def __init__(
        self,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)
        
    def log_sample_images(self, batch, pipeline: FluxPipeline, device: torch.device, step: int):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0).to(device=device)
            generator = [torch.Generator(device=device).manual_seed(seed) for seed in self.sample_seeds]

            sequence = pipeline(
                prompt,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}.png"), image[:, :, ::-1] * 255)
            with open(os.path.join(self.logdir, f"{step}_{idx}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
            for i, img in enumerate(sequence):
                img.save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))

def train(
    logdir: str,
    pretrained_model_path: str,
    train_steps: int = 5000,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 8,
    val_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    checkpointing_steps: int = 10000,
):
    
    args = get_function_args()
    logdir += f"_{get_time_string()}"

    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=gradient_accumulation_steps, gradient_clipping=max_grad_norm, zero_stage=2, offload_optimizer_device='cpu', offload_param_device='cpu')
    # deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=gradient_accumulation_steps, gradient_clipping=max_grad_norm, zero_stage=2) not use cpu offload
    deepspeed_plugin.set_mixed_precision(mixed_precision)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin)
    # accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision) # not use deepspeed

    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder_2 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")

    pipeline = FluxPipeline(
        vae = vae,
        scheduler = scheduler,
        transformer = transformer,
        text_encoder = text_encoder, 
        tokenizer = tokenizer,
        text_encoder_2 = text_encoder_2,
        tokenizer_2 = tokenizer_2,
    )
    pipeline.set_progress_bar_config(disable=True)
    
    train_dataset = SimpleDataset(root="./", mode='train')
    val_dataset = SimpleDataset(root="./", mode='test')
    
    print(train_dataset.__len__(), val_dataset.__len__())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer.enable_gradient_checkpointing()
    
    for model in [vae, text_encoder, text_encoder_2, transformer]:
        model.requires_grad_(False)
    
    trainable_modules = ("transformer_blocks.1", "transformer_blocks.2", "transformer_blocks.3")
    # trainable_modules = ("transformer_blocks", "single_transformer_blocks")
    for name, module in transformer.named_modules():
        if name.startswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        # for params in module.parameters():
        #     params.requires_grad = True

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)
    
    # Use 8-bit Adam for lower memory usage
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = optimizer_class(params_to_optimize, lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleFLUX")

    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)
        
        vae.eval()
        text_encoder.eval()    
        text_encoder_2.eval()
        transformer.train()
        
        # Convert images to latent space
        with torch.no_grad():
            image = batch["image"].to(device=accelerator.device, dtype=weight_dtype)
            model_input = vae.encode(image).latent_dist.sample()
            model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            accelerator.device,
            model_input.dtype,
        )

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input, device=accelerator.device, dtype=model_input.dtype)
        bsz = model_input.shape[0]
        # Sample a random timestep for each image for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching. zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
        # handle guidance
        if unwrap_model(transformer).config.guidance_embeds:
            # guidance_scale = 3.5
            guidance_vec = torch.full((bsz,), 3.5, device=accelerator.device, dtype=model_input.dtype)
        else:
            guidance_vec = None

        # text encoding
        prompt = batch["prompt"]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt=prompt, prompt_2=prompt)

        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        text_ids = text_ids.to(accelerator.device)

        model_pred = transformer(
            hidden_states=packed_noisy_model_input.to(model_input.dtype),
            timestep=timesteps.to(model_input.dtype) / 1000,
            guidance=guidance_vec.to(model_input.dtype),
            pooled_projections=pooled_prompt_embeds.to(model_input.dtype),
            encoder_hidden_states=prompt_embeds.to(model_input.dtype),
            txt_ids=text_ids.to(model_input.dtype),
            img_ids=latent_image_ids.to(model_input.dtype),
            return_dict=False,
        )[0]
        
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        
        # these weighting schemes use a uniform timestep sampling and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        # flow matching loss
        target = noise - model_input
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    transformer.eval()
                    val_batch = next(val_data_yielder)
                    with autocast(dtype=weight_dtype):
                        validation_sample_logger.log_sample_images(
                            batch=val_batch,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                if step % checkpointing_steps == 0:
                    pipeline_save = FluxPipeline(
                        vae=vae,
                        scheduler=scheduler,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        transformer=accelerator.unwrap_model(transformer),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()

if __name__ == "__main__":
    config = "./config.yml"
    train(**OmegaConf.load(config))

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py