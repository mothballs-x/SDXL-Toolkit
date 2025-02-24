# ✨🔮Stable Diffusion ToolKit🔮✨


A **Colab Notebook** that tries the bridge the gap between the convenience of a tool like Automatic1111 and the versatility of a purely script-based notebook. Mostly made for myself, thought it might be somewhat useful to others.

#### Includes:
	- Txt2Img Pipeline
	- Img2Img Pipeline 
	- REALESRGAN Upscaler
	- HiRes Fix
	- GFPGAN Upscaler
	- Lora and Prompt Managers

---

### Setup:

	1. Save utilities/ in Drive root directory
	2. Load Notebook

That's basically it!

---

### Models and LoRAs:

Models and LoRAs can be pulled from Drive, huggingface, or civitai (via API). Use **models.json** and **lora_list.json** to create a directory of file references. Here's the structure:

#### models.json

```json
{
  "model_type": {
    "model_version1": "https://link.to.civitai/api",
    "model_version2": "/content/drive/My Drive/local_model.safetensors",
    "model_version3": "huggingface/repo"
  }
}
```

#### lora_list.json

```json
{
  "name": "lora",
  "link": "https://link.to.civitai",
  "triggers": "These, are, lora, trigger, tags",
  "model": "base/pony/etc..."
}
```
  
It's a rather useful way to manage models/LoRAs without having to take up valuable Drive space. I've added a smattering of models and LoRAs to start out with (**warning: many included LoRAs are NSFW!**)

### Managers:

There are three classes that do most of the work here: LoraManager, Prompter, and ImageGenerator. 

**LoraManager** handles adding/removing LoRAs and modifying weights with a pretty bare-bones UI.

**Prompter** compiles pos/neg prompts with the structure:

	- <embedding tokens> <pony tags?> <'initial' tags> <main prompt>

where *embedding tokens* are added automatically, *pony tags* are optional 'score_9, etc...' tags, *initial prompts* are tags you want to persist over different prompts, and *main prompt* is...your main prompt. There's also an option to either automatically include random danbooru tags in your prompt, or just generate a list of tags to sprinkle in when your creativity is lacking.

**ImageGenerator** handles methods for generation and upscaling, with a configuration collections.dataset that's controlled by widgets. 

### Note on Image2Image generation

I've been have issues with img2img generation overloading the GPU, so I'm trying to figure out how to best manage this. Going to create a branch to work on it. So currently only txt2img and the upscaling methods are fully functional.  




