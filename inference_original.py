import torch
from diffusers import FluxPipeline

# it takes approximately 24G GPU memory to inference in bf16
pretrained_path = './ckpt/FLUX.1-dev' # or directly use black-forest-labs/FLUX.1-dev to download from huggingface
pipe = FluxPipeline.from_pretrained(pretrained_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
