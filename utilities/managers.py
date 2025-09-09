import subprocess
from typing import Union

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


# -----------------------------
# Prompt Manager
# -----------------------------
class PromptManager:
    pony_pos = 'score_9, score_8_up, score_7_up, score_6_up, rating_explicit'
    pony_neg = 'score_5, score_4, score_3, score_2, rating_safe'

    def __init__(self, pipeline, initial_tags=None, pos_tokens=None, neg_tokens=None, df=None, pony=True):
        self.pipeline = pipeline
        self.initial = initial_tags
        self.pos_tokens = ', '.join(pos_tokens) if isinstance(pos_tokens, list) else pos_tokens
        self.neg_tokens = ', '.join(neg_tokens) if isinstance(neg_tokens, list) else neg_tokens
        self.pony = pony
        self.pos_prompt = ""
        self.neg_prompt = ""
        self.df = df

        # Use the pipeline's device (you ensure pipe.to("cuda") elsewhere)
        self.device = getattr(pipeline, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Make sure both text encoders live on the same device as the pipeline
        pipeline.text_encoder.to(self.device).eval()
        pipeline.text_encoder_2.to(self.device).eval()
        self.enc_dtype = pipeline.text_encoder.dtype  # keep dtype in sync

        # Compel on the same device as the encoders/pipeline
        self.compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.device,  # critical: make token ids match encoder weights
        )

    def randan(self, count=15, threshold=1):
        if self.df is None or len(self.df) == 0:
            return ""
        tags = []
        while len(tags) < count:
            x = random.randint(0, len(self.df) - 1)
            row = self.df.iloc[x]
            tag = re.sub(r'_', ' ', str(row[0]))
            tag_number = float(row[1])
            if tag_number > threshold:
                tags.append(tag)
        return ', '.join(tags)

    def create_prompt(self, main_pos=None, main_neg=None, rand_tags=False, shuffle=False):
        # Build lists
        pos_parts, neg_parts = [], []
        if self.pos_tokens: pos_parts.append(self.pos_tokens)
        if self.neg_tokens: neg_parts.append(self.neg_tokens)
        if self.pony:
            pos_parts.append(self.pony_pos)
            neg_parts.append(self.pony_neg)
        if self.initial: pos_parts.append(self.initial)
        if main_pos: pos_parts.append(main_pos)
        if main_neg: neg_parts.append(main_neg)
        if rand_tags: pos_parts.append(self.randan())

        # Clean and optionally shuffle
        pos_parts = [re.sub(r'\n', '', t) for t in pos_parts]
        neg_parts = [re.sub(r'\n', '', t) for t in neg_parts]
        if shuffle:
            random.shuffle(pos_parts)

        # Persist strings (nice for UI)
        self.pos_prompt = ", ".join([t for t in pos_parts if t])
        self.neg_prompt = ", ".join([t for t in neg_parts if t])

        # -------- Conditional embeddings via Compel (on self.device) --------
        # Batched order: [positive, negative]
        prompt_embeds = self.compel([self.pos_prompt or "", self.neg_prompt or ""])

        # -------- Pooled embeddings via encoder_2 (on same device) ----------
        # Tokenize with tokenizer_2 and move tokens to the *actual* device of encoder_2
        enc2 = self.compel.text_encoder[1]
        enc2_dev = next(enc2.parameters()).device  # should equal self.device, but read it to be future-proof
        tok2 = self.compel.tokenizer[1](
            [self.pos_prompt or "", self.neg_prompt or ""],
            padding="max_length",
            max_length=self.compel.tokenizer[1].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok2 = {k: v.to(enc2_dev) for k, v in tok2.items()}

        with torch.no_grad():
            out2 = enc2(
                tok2["input_ids"],
                attention_mask=tok2.get("attention_mask", None),
                output_hidden_states=False,
            )
        # Handle both CLIP variants
        if hasattr(out2, "text_embeds"):       # CLIPTextModelWithProjection
            pooled_embeds = out2.text_embeds
        elif hasattr(out2, "pooler_output"):
            pooled_embeds = out2.pooler_output
        else:                                  # Fallback: CLS token
            pooled_embeds = out2.last_hidden_state[:, 0, :]

        # Ensure both tensors are exactly on the pipeline device/dtype
        def _to(t):
            if isinstance(t, (list, tuple)):
                return [x.to(self.device, dtype=self.enc_dtype) for x in t]
            return t.to(self.device, dtype=self.enc_dtype)

        prompt_embeds = _to(prompt_embeds)
        pooled_embeds = _to(pooled_embeds)

        return (prompt_embeds, pooled_embeds)


# -----------------------------
# LoRA Manager
# -----------------------------
class LoraManager:

    def __init__(self, pipeline, civitai_token, lora_dir: str = "/content/loras"):
        self.pipeline = pipeline
        self.lora_dir = Path(lora_dir)  # Convert to Path object
        self.lora_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.civitai_token = civitai_token
        self.loras = {}

    class Lora:
        def __init__(self, container, path: Path, filename: str, name: str, weight: float = 0.0):
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
            return f'Lora({self.name, self.weight})'

    def update_weights(self):
        """Apply updated weights to LoRAs in the pipeline"""
        lora_names = [lora.name for lora in self.loras.values()]
        lora_weights = [lora.weight for lora in self.loras.values()]
        if lora_names:
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
        if isinstance(names, str):
            names = [names]
        elif not isinstance(names, list):
            raise TypeError("Names must be a string or a list of strings")

        # Remove adapters from pipeline
        self.pipeline.delete_adapters(names)

        # Clean internal map
        for name in names:
            if name in self.loras:
                del self.loras[name]
            else:
                print(f"[Warning] LoRA '{name}' not found.")
        self.update_weights()

    def list_loras(self):
        """Return all available LoRAs"""
        return {lora.name: lora.weight for lora in self.loras.values()}

    def clear_loras(self):
        """Remove all LoRAs"""
        self.loras = {}
        self.update_weights()

    def get_lora_from_link(self, model_code: str, name: str):
        """Download LoRAs from CivitAI and save as safetensors."""
        lora_url = f"https://civitai.com/api/download/models/{model_code}?token={self.civitai_token}"
        with requests.get(lora_url, stream=True) as resp:
            if resp.status_code != 200:
                raise ValueError(f"Failed to download LoRA '{name}'. Status: {resp.status_code} - {resp.text}")

            content_disp = resp.headers.get("content-disposition", "")
            if "filename=" in content_disp:
                file_name = content_disp.split("filename=")[1].strip('"')
            else:
                file_name = f"{name}.safetensors"  # Fallback name

            lora_path = self.lora_dir / file_name
            with open(lora_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

        if lora_path.suffix.lower() != ".safetensors":
            print(f"⚠️ Downloaded file does not have a .safetensors extension: {lora_path}")

        return lora_path


# -----------------------------
# Image Generation
# -----------------------------
@dataclass
class Config:
    width: int = 768
    height: int = 768
    num_imgs: int = 4
    steps: tuple = (29, 29)  # Ensure these are integers when used
    cfg: float = 8.0
    scale: float = 1.0
    strength: float = 0.0  # img2img strength
    current_seed: int = None
    clip_skip: int = 1

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


class ImageGenerator:
    def __init__(self, pipeline, upscaler):
        self.pipeline = pipeline
        self.upscaler = upscaler  # function/callable
        self.config = Config()

    def txt2img(self, prompt, seed=None):
        if not isinstance(prompt, tuple):
            raise TypeError('Prompt must be a tuple of conditional and pooled embeddings')

        generator = torch.Generator(device=self.pipeline.device)

        if seed is None:
            seed = generator.seed()
        else:
            generator.manual_seed(seed)
            print(f'Seed: {seed}')

        images = self.pipeline(
            prompt_embeds=prompt[0][0:1],
            pooled_prompt_embeds=prompt[1][0:1],
            negative_prompt_embeds=prompt[0][1:2],
            negative_pooled_prompt_embeds=prompt[1][1:2],
            num_images_per_prompt=self.config.num_imgs,
            generator=generator,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=int(self.config.steps[0]),
            guidance_scale=self.config.cfg,
            clip_skip=self.config.clip_skip,
        ).images

        self.config.current_seed = seed
        return images

    def upscale(self, images):
        return self.upscaler(
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

        generator = torch.Generator(device=self.pipeline.device)
        if seed is None:
            seed = torch.seed()
            generator.manual_seed(seed)
            print(f'Seed: {seed}')
        else:
            generator.manual_seed(seed)

        num_images = self.config.num_imgs if not for_hires else 1

        output = self.pipeline(
            prompt_embeds=prompt[0][0:1],
            pooled_prompt_embeds=prompt[1][0:1],
            negative_prompt_embeds=prompt[0][1:2],
            negative_pooled_prompt_embeds=prompt[1][1:2],
            num_inference_steps=int(self.config.steps[1]),
            image=image,
            width=self.config.width,
            height=self.config.height,
            num_images_per_prompt=num_images,
            generator=generator,
            strength=self.config.strength,
        ).images
        return output

    def hi_res(self, prompt, scale=None):
        if not isinstance(prompt, tuple):
            raise TypeError('Prompt must be a tuple of conditional and pooled embeddings')

        if scale is not None:
            self.config.scale = scale

        first_pass_images = self.txt2img(prompt)

        self.pipeline.enable_vae_tiling()

        original_width = self.config.width
        original_height = self.config.height

        # If you intend to *render* larger in the second pass, adjust here;
        # but note you're feeding upscaled images into img2img anyway:
        self.config.width = int(self.config.width * self.config.scale)
        self.config.height = int(self.config.height * self.config.scale)

        upscaled_images = self.upscale(first_pass_images)

        seeds = [self.config.current_seed + n for n in range(len(upscaled_images))]
        results = []
        for i, img in enumerate(upscaled_images):
            local_generator = torch.Generator(device=self.pipeline.device)
            local_generator.manual_seed(seeds[i])
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

        subprocess.run(process, check=False)

        outputs = [Image.open(output) for output in Path('/content/upscaled/restored_imgs').glob('*')]
        return outputs