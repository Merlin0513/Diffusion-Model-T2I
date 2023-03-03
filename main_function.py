import torch
import importlib
import time
import numpy as np
import argparse, os, sys

from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from contextlib import nullcontext

from torch import autocast
from contextlib import contextmanager, nullcontext


#from method.ddim import DDIMSampler
from ldm.models.diffusion.ddim import DDIMSampler
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Define function to load model from configs and import them:
def import_model(string):
    target, c = string.rsplit(".", 1)
    return getattr(importlib.import_module(target, package=None), c)

# Part I. Model Settings:

# def get_device():
#     return 'mps' if(torch.backends.mps.is_available()) else return 'cpu'
device = torch.device('mps')

# Generate and define output result path
os.makedirs('outputs/txt2img', exist_ok = True)
outpath = 'outputs/txt2img'

# Set output Height: H, Width: W, Number of Channels: C, Downsampling Factor: F
H, W, C, F = 1024, 1024, 4, 8
shape = [C, H // F, W // F]

# According to the DDIM paper, all the dataset can use 'precision_scope == nullcontext'
# We can also try 'autocast' if I put it in main function.
precision_scope = nullcontext


# Load configs of model from bfirsh github, which is the same model in OpenAI text-to-image model
config = OmegaConf.load("config/v1-inference.yaml")

# Import model from config using import_model function
model = import_model(config.model["target"])(**config.model.get("params", dict()))

# Put model on Mac M1 GPU 'MPS'
model.to('mps')
model.eval()

# Load diffusion model from Method part in my report
sampler = DDIMSampler(model)

# Define DDIM steps
ddim_steps = 50

# Set default batch_size
batch_size = 3
#n_rows = batch_size
scale = 7.5

# Part II. Running Model:

# Define functions to show sampling process using Method/DDIM
def generate_samples(model, data):
    uncond_meaning = model.get_learned_conditioning(batch_size * ['']) if scale != 1.0 else None
    cond_meaning = model.get_learned_conditioning(data)
    
    samples, _ = sampler.sample(
        S=ddim_steps, conditioning=cond_meaning, batch_size=batch_size,
        shape=shape, unconditional_guidance_scale=scale, 
        unconditional_conditioning=uncond)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_text", nargs="?", type=str)
    # Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    parser.add_argument("--scale:", type=float, default = 7.5)
    result = parser.parse_args()
    
    input_text = result.input_text
    data = [batch_size * [input_text]]
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                all_samples = list()
                n_iter = 1
                for _ in range(n_iter):
                # Here I set n_iter as default == 1 since limited GPU resources.
                # The first stage of learning the context
                    image = generate_samples(model, data = data)
                    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(image.astype(np.uint8))
                    img.save(os.path.join(outpath, name = f'{time.strftime("%Y%m%d_%H%M%S")}.png'))
                
    
if __name__ == "__main__":
    main()