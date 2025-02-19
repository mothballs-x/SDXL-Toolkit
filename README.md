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

**ImageGenerator** handles methods for generation and upscaling, with a configuration dataset that's controlled by widgets. 

Class definitions and helper functions for upscaling, text-inversion, and changing schedulers are included in the script itself. The idea is to have a base to work off of that you can modify on the fly; if you want to just run basic generations, use the UI, otherwise the code is there and relatively easy to modify. The rest of the utilities modules are for non-generation-related functionality: logging, saving, viewing, etc.

**Basically**, just a notebook I made for myself to learn how to use diffusers, and figured, "hey, maybe someone out there might find this useful"! 👽✌️ Would welcome any and all improvements.

#### Note on HiRes Generation:

This is still somewhat of a work in progress, as results vary depending on the model. The key seems to be in finding the noise strength sweet spot but there doesn't seem to be very much margin for error. But if you're looking for figures made up of little peices of other human figures, then this might be right up your alley!





