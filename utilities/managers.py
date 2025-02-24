import subprocess
from typing import Union

import numpy as np
# PromptManager Imports
from compel import Compel, ReturnedEmbeddingsType
import random
import re

# LoraManager
import requests
from pathlib import Path
from PIL import Image

# Image Generator
from dataclasses import dataclass
import torch


# Class for managing prompts
class PromptManager:

    pony_pos = 'score_9, score_8_up, score_7_up, score_6_up, rating_explicit'
    pony_neg = 'score_5, score_4, score_3, score_2, rating_safe'

    def __init__(
      self, pipeline,
      initial_tags=None,
      pos_tokens=None, neg_tokens=None,
      df=None, pony=True
      ):

        self.initial = initial_tags
        pos_tokens = ', '.join(pos_tokens) if isinstance(pos_tokens, list) else pos_tokens
        neg_tokens = ', '.join(neg_tokens) if isinstance(neg_tokens, list) else neg_tokens
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.pony = pony

        self.prompt = ""
        self.pos_prompt = ""
        self.neg_prompt = ""

        self.df = df

        self.compel = Compel(
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
        )

    def randan(self, count=15, threshold=1):
        tags = []

        while len(tags) < count:  # Ensure we always get "count" valid tags
            x = random.randint(0, len(self.df) - 1)  # ✅ Avoid out-of-bounds indexing

            tag, tag_number = self.df.loc[x].values
            tag = re.sub(r'_', ' ', tag)

            if tag_number > threshold:  # ✅ Drop tags that don't meet the threshold
                tags.append(tag)
            # print(tag, end=', ')  # ✅ Shows selected tags in real-time

            # print('\n')
        return ', '.join(tags)

    def create_prompt(self, main_pos=None, main_neg=None, rand_tags=False, shuffle=False):
        # Reset pos_prompt and neg_prompt before constructing
        self.pos_prompt = []
        self.neg_prompt = []

        if self.pos_tokens:
            self.pos_prompt.append(self.pos_tokens)
        if self.neg_tokens:
            self.neg_prompt.append(self.neg_tokens)
        if self.pony:
            self.pos_prompt.append(self.pony_pos)
            self.neg_prompt.append(self.pony_neg)
        if self.initial:
            self.pos_prompt.append(self.initial)
        if main_pos:
            self.pos_prompt.append(main_pos)
        if main_neg:
            self.neg_prompt.append(main_neg)
        if rand_tags:
            self.pos_prompt.append(self.randan())

        self.pos_prompt = [re.sub(r'\n', '', tag) for tag in self.pos_prompt]
        self.neg_prompt = [re.sub(r'\n', '', tag) for tag in self.neg_prompt]

        if shuffle:
            random.shuffle(self.pos_prompt)

        self.pos_prompt = ", ".join(self.pos_prompt)
        self.neg_prompt = ", ".join(self.neg_prompt)

        prompt_embeddings = self.compel([self.pos_prompt, self.neg_prompt])

        return prompt_embeddings


# Class for managing LoRAs
class LoraManager:

    def __init__(self, pipeline, civitai_token, lora_dir: str = "/content/loras"):
        self.pipeline = pipeline
        self.lora_dir = Path(lora_dir)  # Convert to Path object
        self.lora_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.civitai_token = civitai_token
        self.loras = {}

    class Lora:
        def __init__(self, container, path: str, filename: str, name: str, weight: float = 0.0):
            self.container = container
            self.path = path
            self.filename = filename
            self.name = name
            self.weight = float(weight)
            self.load_lora()

        def change_weight(self, weight: float):
            self.weight = float(weight)
            self.container.update_weights()

        def load_lora(self):
            """Load LoRA weights into pipeline"""
            self.container.pipeline.load_lora_weights(
                self.path,
                adapter_name=self.name
                )

        def __repr__(self):
            return f'Lora({self.name, self.weight}'

    def update_weights(self):
        """Apply updated weights to LoRAs in the pipeline"""
        lora_names = [lora.name for lora in self.loras.values()]
        lora_weights = [lora.weight for lora in self.loras.values()]
        self.pipeline.set_adapters(lora_names, lora_weights)

    def add_lora(self, path: str, name: str, weight: float = 0.0):
        """Add a LoRA either from a local file or from CivitAI"""
        if path.startswith("http"):
            model_number = path.split('/')[-1].split('?')[0]
            path = self.get_lora_from_link(model_number, name)  # Download LoRA from link
        else:
            path = Path(path)

        if path.exists():
            self.loras[name] = LoraManager.Lora(self, path, path.name, name, weight)
            self.update_weights()
        else:
            print(f"⚠️ LoRA path '{path}' does not exist.")

    def delete_lora(self, names: Union[str, list]):
        """Remove a LoRA from the manager"""
        if not (isinstance(names, list) or isinstance(names, str)):
            raise TypeError("Names must be a string or a list of strings")
        if isinstance(names, str):
            names = [names]
        self.pipeline.delete_adapters(names)
        for name in names:
            if name in self.loras:
                del self.loras[name]
                self.update_weights()
            else:
                print(f"[Warning] LoRA '{name}' not found.")

    def list_loras(self):
        """Return all available LoRAs"""
        return {lora.name: lora.weight for lora in self.loras.values()}

    def clear_loras(self):
        """Remove all LoRAs"""
        self.loras = {}
        self.update_weights()

    def get_lora_from_link(self, model_code: str, name: str):
        """Download LoRA from CivitAI and ensure it's saved as a binary safetensors file."""
        lora_url = f"https://civitai.com/api/download/models/{model_code}?token={self.civitai_token}"
        resp = requests.get(lora_url, stream=True)

        if resp.status_code != 200:
            raise ValueError(f"Failed to download LoRA '{name}'. Status: {resp.status_code} - {resp.text}")

        content_disp = resp.headers.get("content-disposition", "")
        if "filename=" in content_disp:
            file_name = content_disp.split("filename=")[1].strip('"')
        else:
            file_name = f"{name}.safetensors"  # Fallback name

        lora_path = self.lora_dir / file_name
        with open(lora_path, "wb") as f:
            f.write(resp.content)  # Ensure binary mode is used

        if not lora_path.suffix == ".safetensors":
            print(f"⚠️ Downloaded file does not have a .safetensors extension: {lora_path}")

        return lora_path


@dataclass
class Config:
    width: int = 768
    height: int = 768
    num_imgs: int = 4
    steps: tuple = (29, 29)  # Ensure these are integers when used
    cfg: float = 8.0
    scale: float = 1.0
    strength: float = 0.0  # Fixed typo from 'strength'
    current_seed: int = None
    clip_skip: int = 1

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


class ImageGenerator:
    def __init__(self, pipeline, upscaler):
        self.pipeline = pipeline
        self.upscale = upscaler  # Fixed incorrect reference
        self.config = Config()

    def txt2img(self, prompt, seed=None):
        # Reset U-Net state if applicable

        if not isinstance(prompt, tuple):
            raise TypeError('Prompt must be a tuple of conditional and pooled embeddings')

        generator = torch.Generator(device=self.pipeline.device)

        if seed is None:
            seed = generator.seed()
        else:
            generator.manual_seed(seed)
            print(f'Seed: {seed}')

        # Create noise in latent space
        latents = torch.randn((1, 4,                    # Change 1 to self.config.num_imgs if batching
                               self.config.height // 8,
                               self.config.width // 8
                               )).to(self.pipeline.device)

        noisy_image = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor
        ).sample

        noisy_image = (noisy_image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
        noisy_image = noisy_image.cpu().permute(0, 2, 3, 1).numpy()  # Convert (B, C, H, W) → (B, H, W, C)

        pil_images = [Image.fromarray((img * 255).astype("uint8")) for img in noisy_image]

        print(f'noise dtype: {noisy_image.dtype}')

        strength = 1.0

        images = self.pipeline(
            # Prompt
            prompt_embeds=prompt[0][0:1],
            pooled_prompt_embeds=prompt[1][0:1],
            negative_prompt_embeds=prompt[0][1:2],
            negative_pooled_prompt_embeds=prompt[1][1:2],

            # img2img settings
            strength=strength,
            image=pil_images,

            # Generation Settings
            num_images_per_prompt=1,
            generator=generator,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=self.config.steps[0],  # Use first step value
            guidance_scale=self.config.cfg,
            clip_skip=self.config.clip_skip,
        ).images

        self.config.current_seed = seed

        return images

    def upscale(self, images):

        return self.upscale(
            images,
            model_name='RealESRGAN_x4plus',
            scale_factor=self.config.scale,
            half_precision=False,
            tile=800,
        )

    def img2img(self, image, prompt, seed=None, for_hires=False):

        if image.size != (1024, 1024):
            print('Image must be 1024x1024 to avoid distortions...')

        if not isinstance(prompt, tuple):
            raise TypeError('Prompt must be a tuple of conditional and pooled embeddings')

        if seed is None:
            generator = torch.Generator(device=self.pipeline.device)
            seed = torch.seed()
            generator.manual_seed(seed)
            print(f'Seed: {seed}')
        else:
            generator = torch.Generator(device=self.pipeline.device)
            generator.manual_seed(seed)

        num_images = self.config.num_imgs if not for_hires else 1

        output = self.pipeline(  # Fixed incorrect `img2img_pipe`
            prompt_embeds=prompt[0][0:1],
            pooled_prompt_embeds=prompt[1][0:1],
            negative_prompt_embeds=prompt[0][1:2],
            negative_pooled_prompt_embeds=prompt[1][1:2],
            num_inference_steps=self.config.steps[1],  # Second step value
            image=image,
            width=self.config.width,
            height=self.config.height,
            num_images_per_prompt=num_images,
            generator=generator,
            strength=self.config.strength,  # Fixed typo
        ).images
        return output

    def hi_res(self, prompt, scale=None):
        if not isinstance(prompt, tuple):
            raise TypeError('Prompt must be a tuple of conditional and pooled embeddings')

        if scale is None:
            self.config.scale = scale

        first_pass_images = self.txt2img(prompt)

        self.pipeline.enable_vae_tiling()  # Fixed incorrect `img2img_pipe`

        original_width = self.config.width
        original_height = self.config.height

        self.config.width = int(self.config.width * self.config.scale)  # Fixed missing SCALE
        self.config.height = int(self.config.height * self.config.scale)

        upscaled_images = self.upscale(first_pass_images, scale=self.config.scale)

        seeds = [self.config.current_seed + n for n in range(len(upscaled_images))]

        results = []
        for i, img in enumerate(upscaled_images):
            local_generator = torch.Generator(device=self.pipeline.device)
            local_generator.manual_seed(seeds[i])  # Fixed seed usage
            output = self.img2img(img, prompt, seed=seeds[i], for_hires=True)
            results.append(output[0])

        self.config.height = original_height
        self.config.width = original_width

        return results

    def gfpgan(self, images, scale=None):

        if scale is None:
            scale = self.config.scale
        gfpgan_script = '/content/GFPGAN/inference_gfpgan.py'

        if not Path(gfpgan_script).exists():
            raise OSError("gfpgan has not been installed correctly on the system")

        if not isinstance(images, list):
            images = [images]
        for i, img in enumerate(images, 1):
            img.save(f'/content/upscale/to_upscale{i:02}.png')

        process = ["python3",
                   '/content/GFPGAN/inference_gfpgan.py',
                   "-i", "/content/upscale",
                   "-o", "/content/upscaled",
                   "-v", "1.3",
                   "-s", str(scale),
                   ]

        subprocess.run(process)

        outputs = [Image.open(output) for output in Path('/content/upscaled/restored_imgs').glob('*')]

        return outputs
