import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content/Open-Sora-Plan-v1.0.0-hf')
os.chdir('/content/Open-Sora-Plan-v1.0.0-hf')

import random
import imageio
import torch
from diffusers import PNDMScheduler
from datetime import datetime
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.sample.pipeline_videogen import VideoGenPipeline

model_path = 'LanguageBind/Open-Sora-Plan-v1.0.0'
ae = 'CausalVAEModel_4x8x8'
force_images = False
text_encoder_name = 'DeepFloyd/t5-v1_1-xxl'
version = '65x512x512'

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, 203279)
    return seed

def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

def generate_img(prompt, sample_steps, scale, seed=0, randomize_seed=False, force_images=False, transformer_model=None, videogen_pipeline=None):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)
    video_length = transformer_model.config.video_length if not force_images else 1
    height, width = int(version.split('x')[1]), int(version.split('x')[2])
    num_frames = 1 if video_length == 1 else int(version.split('x')[0])
    with torch.no_grad():
        videos = videogen_pipeline(prompt,
                                   video_length=video_length,
                                   height=height,
                                   width=width,
                                   num_inference_steps=sample_steps,
                                   guidance_scale=scale,
                                   enable_temporal_attentions=not force_images,
                                   num_images_per_prompt=1,
                                   mask_feature=True,
                                   ).video

    torch.cuda.empty_cache()
    videos = videos[0]
    tmp_save_path = 'tmp.mp4'
    imageio.mimwrite(tmp_save_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
    display_model_info = f"Video size: {num_frames}×{height}×{width}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
    return tmp_save_path, prompt, display_model_info, seed

class Predictor(BasePredictor):
    def setup(self) -> None:
        device = torch.device('cuda:0')
        self.transformer_model = LatteT2V.from_pretrained(model_path, subfolder=version, torch_dtype=torch.float16, cache_dir='cache_dir').to(device)
        vae = getae_wrapper(ae)(model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
        vae.vae.enable_tiling()
        image_size = int(version.split('x')[1])
        latent_size = (image_size // ae_stride_config[ae][1], image_size // ae_stride_config[ae][2])
        vae.latent_size = latent_size
        self.transformer_model.force_images = force_images
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir", torch_dtype=torch.float16).to(device)

        self.transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        self.videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=self.transformer_model).to(device=device)
    def predict(
        self,
        prompt: str = Input(default="A quiet beach at dawn, the waves gently lapping at the shore and the sky painted in pastel hues."),
        sample_steps: int = Input(default=50),
        guidance_scale: float = Input(default=10.0),
        seed: int = Input(default=0),
        randomize_seed: bool = True,
        force_images: bool = False
    ) -> Path:
        tmp_save_path, prompt, display_model_info, seed = generate_img(prompt=prompt, sample_steps=sample_steps, scale=guidance_scale, seed=seed, randomize_seed=randomize_seed, force_images=force_images, transformer_model=self.transformer_model, videogen_pipeline=self.videogen_pipeline)
        return Path(tmp_save_path)