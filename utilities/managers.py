import subprocess
from typing import Union

# PromptManager Imports
from compel import Compel, ReturnedEmbeddingsType
import random
import re

# LoRA Manager
import requests
from pathlib import Path
from PIL import Image

# Image Generator
from dataclasses import dataclass
import torch


# -----------------------------
# Prompt Manager (GPU-only embeddings)
# -----------------------------
class PromptManager:
    """
    Builds SDXL prompt and negative prompt strings, then produces
    (cond_embeds_batch, pooled_embeds_batch) with batch order [pos, neg],
    entirely on the pipeline's device (GPU). No CPU/offload during encoding.
    """

    pony_pos = 'score_9, score_8_up, score_7_up, score_6_up, rating_explicit'
    pony_neg = 'score_5, score_4, score_3, score_2, rating_safe'

    def __init__(
        self, pipeline,
        initial_tags=None,
        pos_tokens=None, neg_tokens=None,
        df=None, pony=True
    ):
        self.pipeline = pipeline

        # Persisted tokens/options
        self.initial = initial_tags
        pos_tokens = ', '.join(pos_tokens) if isinstance(pos_tokens, list) else pos_tokens
        neg_tokens = ', '.join(neg_tokens) if isinstance(neg_tokens, list) else neg_tokens
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.pony = pony

        # Human-readable strings for last-built prompts (nice for UI)
        self.pos_prompt = ""
        self.neg_prompt = ""

        # Optional DataFrame for random tags
        self.df = df

        # Device & dtype policy: all on pipeline's device (GPU), no CPU hops
        self.device = getattr(pipeline, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError(
                f"PromptManager expects the pipeline to be on GPU, but got {self.device}. "
                "Call `pipe.to('cuda')` before constructing PromptManager."
            )

        # Pin both text encoders to the pipeline device (no offloading)
        pipeline.text_encoder.to(self.device).eval()
        pipeline.text_encoder_2.to(self.device).eval()

        # Use the encoders' dtype (UNet will often be fp16 on GPU)
        self.enc_dtype = getattr(pipeline.unet, "dtype", pipeline.text_encoder.dtype)

        # Compel on the same device so token indices match encoder weights
        self.compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.device,  # critical: generate token IDs on the same device
        )

    def randan(self, count=15, threshold=1):
        """Return `count` tags with tag_number > threshold."""
        if self.df is None or len(self.df) == 0:
            return ""
        tags = []
        # Prefer iloc to avoid surprises with non-integer index
        while len(tags) < count:
            x = random.randint(0, len(self.df) - 1)
            row = self.df.iloc[x]
            tag = re.sub(r'_', ' ', str(row[0]))
            tag_number = float(row[1])
            if tag_number > threshold:
                tags.append(tag)
        return ', '.join(tags)

    def _build_prompt_strings(self, main_pos=None, main_neg=None, rand_tags=False, shuffle=False):
        """Assemble positive/negative prompt strings from parts."""
        pos_parts, neg_parts = [], []

        if self.pos_tokens:
            pos_parts.append(self.pos_tokens)
        if self.neg_tokens:
            neg_parts.append(self.neg_tokens)
        if self.pony:
            pos_parts.append(self.pony_pos)
            neg_parts.append(self.pony_neg)
        if self.initial:
            pos_parts.append(self.initial)
        if main_pos:
            pos_parts.append(main_pos)
        if main_neg:
            neg_parts.append(main_neg)
        if rand_tags:
            pos_parts.append(self.randan())

        # Clean and optionally shuffle
        pos_parts = [re.sub(r'\n', '', t) for t in pos_parts]
        neg_parts = [re.sub(r'\n', '', t) for t in neg_parts]
        if shuffle:
            random.shuffle(pos_parts)

        # Persist strings (for UI/debug) and return
        self.pos_prompt = ", ".join([t for t in pos_parts if t])
        self.neg_prompt = ", ".join([t for t in neg_parts if t])
        return self.pos_prompt or "", self.neg_prompt or ""

    def create_prompt(self, main_pos=None, main_neg=None, rand_tags=False, shuffle=False):
        """
        Returns (cond_embeds_batch, pooled_embeds_batch), both on the pipeline device/dtype.
        Batch order: [positive, negative].
        """
        pos_str, neg_str = self._build_prompt_strings(main_pos, main_neg, rand_tags, shuffle)

        # -------- Conditional embeddings via Compel (GPU) --------
        # Some Compel versions may return either a single tensor or (cond, pooled).
        cond_raw = self.compel([pos_str, neg_str])

        if isinstance(cond_raw, (list, tuple)):
            # If a tuple came back, first element must be conditional embeddings
            cond = cond_raw[0]
        else:
            cond = cond_raw

        # Ensure cond ends up exactly on pipeline device/dtype
        cond = cond.to(self.device, dtype=self.enc_dtype)

        # -------- Pooled embeddings via encoder_2 (GPU) ----------
        enc2 = self.pipeline.text_encoder_2
        enc2_dev = next(enc2.parameters()).device  # should be the same as self.device

        tok2 = self.pipeline.tokenizer_2(
            [pos_str, neg_str],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
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

        if hasattr(out2, "text_embeds"):            # CLIPTextModelWithProjection
            pooled = out2.text_embeds
        elif hasattr(out2, "pooler_output"):        # some variants
            pooled = out2.pooler_output
        else:                                       # Fallback: CLS token
            pooled = out2.last_hidden_state[:, 0, :]

        pooled = pooled.to(self.device, dtype=self.enc_dtype)

        return (cond, pooled)

    # --- Extended-prompt path (A1111-style prompt chunking) -------------------
    def _chunk_by_tokens(self, text: str, tokenizer, max_len: int):
        """Split `text` into chunks that each fit within `max_len` tokens for `tokenizer`.
        Keeps delimiters so we don't smash words together.
        """
        import re as _re
        parts = _re.split(r'(\s+|,)', text)
        chunks, cur = [], []
        for p in parts:
            if p == '':
                continue
            tentative = ''.join(cur + [p])
            ids = tokenizer([tentative], padding=False, truncation=False, return_tensors='pt')['input_ids']
            if ids.shape[-1] <= max_len:
                cur.append(p)
            else:
                if cur:
                    chunks.append(''.join(cur).strip(', '))
                    cur = [p]
                else:
                    chunks.append(p)
                    cur = []
        if cur:
            chunks.append(''.join(cur).strip(', '))
        return [c.strip() for c in chunks if c.strip()]

    def _encode_chunks_compel(self, chunks):
        """Encode a list of chunk strings with Compel and concatenate along seq_len."""
        embeds = []
        for ch in chunks:
            e = self.compel([ch])
            if isinstance(e, (list, tuple)):
                e = e[0]
            embeds.append(e.to(self.device, dtype=self.enc_dtype))
        return torch.cat(embeds, dim=1) if len(embeds) > 1 else embeds[0]

    def _pooled_for_chunks(self, chunks):
        """Compute a pooled embedding for SDXL by averaging encoder_2 pooled outputs,
        weighted by token counts per chunk.
        """
        tok2 = self.pipeline.tokenizer_2
        enc2 = self.pipeline.text_encoder_2
        enc2_dev = next(enc2.parameters()).device

        pooled_sum = None
        weight_sum = 0.0

        for ch in chunks:
            batch = tok2([ch], padding='max_length',
                         max_length=tok2.model_max_length,
                         truncation=True, return_tensors='pt')
            batch = {k: v.to(enc2_dev) for k, v in batch.items()}
            with torch.no_grad():
                out2 = enc2(batch['input_ids'],
                            attention_mask=batch.get('attention_mask', None),
                            output_hidden_states=False)
            if hasattr(out2, 'text_embeds'):
                pooled = out2.text_embeds
            elif hasattr(out2, 'pooler_output'):
                pooled = out2.pooler_output
            else:
                pooled = out2.last_hidden_state[:, 0, :]

            # approximate weight by true (untruncated) token length
            true_ids = tok2([ch], padding=False, truncation=False, return_tensors='pt')['input_ids']
            w = float(true_ids.shape[-1])
            pooled = pooled.to(self.device, dtype=self.enc_dtype)
            pooled_sum = pooled if pooled_sum is None else pooled_sum + w * pooled
            weight_sum += w

        return pooled_sum / max(weight_sum, 1.0)

    def create_prompt_extended(self, main_pos=None, main_neg=None, rand_tags=False, shuffle=False):
        """Build embeddings using prompt chunking (multi-pass encoding) so prompts
        longer than the encoders' windows still contribute. Returns the same tuple
        as `create_prompt`.
        """
        pos_str, neg_str = self._build_prompt_strings(main_pos, main_neg, rand_tags, shuffle)

        # Chunk using the longer SDXL encoder as sizing reference
        tok2 = self.pipeline.tokenizer_2
        max_len = tok2.model_max_length

        pos_chunks = self._chunk_by_tokens(pos_str, tok2, max_len) if pos_str else [""]
        neg_chunks = self._chunk_by_tokens(neg_str, tok2, max_len) if neg_str else [""]

        pos_cond = self._encode_chunks_compel(pos_chunks)
        neg_cond = self._encode_chunks_compel(neg_chunks)

        # Stack into batch order [pos, neg]
        cond = torch.cat([pos_cond, neg_cond], dim=0).to(self.device, dtype=self.enc_dtype)

        # Pooled embeddings from encoder_2 (weighted average across chunks)
        pos_pooled = self._pooled_for_chunks(pos_chunks)
        neg_pooled = self._pooled_for_chunks(neg_chunks)
        pooled = torch.cat([pos_pooled, neg_pooled], dim=0)

        return (cond, pooled)

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
        """Download LoRA from CivitAI and save as safetensors."""
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
        """
        Expects `prompt` to be a tuple: (cond_batch, pooled_batch), each with batch [pos, neg].
        """
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
        """
        Expects `prompt` to be a tuple: (cond_batch, pooled_batch), each with batch [pos, neg].
        """
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