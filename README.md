# ✨🔮Stable Diffusion ToolKit🔮✨


A **Colab Notebook** that tries the bridge the gap between the convenience of a tool like Automatic1111 and the versatility of a purely script-based notebook. Mostly made for myself, thought it might be somewhat useful to others. For features with asterisks, see note at bottom of readme.

#### Includes:
	- Txt2Img Pipeline
	- Img2Img Pipeline* 
	- REALESRGAN Upscaler
	- HiRes Fix*
	- GFPGAN Upscaler
	- Lora and Prompt Managers
---

### Setup:

	1. Download notebook
	2. Load notebook in colab

That's basically it. There's a line in the notebook that clones the necessary files for running the notebook, which are broken down into utilities for functionality and resources that compile models, LoRAs, and embeddings from civitai, huggingface, or a local directory. 

---

### Models and LoRAs:

Models and LoRAs can be downloaded from Drive, huggingface, or civitai (via API). Use **models.json** and **lora_list.json** to create a directory of file references. Here's the structure for each:

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

#### embeddings.json

```json
{
  "name": {
    "positive": {
      "link": "http://civitai.com/api/download/models/123456",
      "token": "Positive_Embedding_Token"
    },
    "negative": {
      "link": "https://civitai.com/api/download/models/123456",
      "token": "Negative_Embedding_Token"
    }
  }
}
```
For embeddings, both "positive" and "negative" are necessary; just leave "link" and "token" blank in one or the other if you're not loading a pair of embeddings.   
I've found it's a rather useful way to manage models/LoRAs without having to take up drive space. But if you have a locally stored LoRA/embedding you can enter its path/url and the notebook will process it.

### Managers:

There are three classes that do most of the work here: LoraManager, Prompter, and ImageGenerator. 

**LoraManager** simplifies adding/removing LoRAs and modifying their weights.

**PromptManager** compiles pos/neg prompts using the following structure:

	- <embedding tokens> <pony tags?> <'initial' tags> <main prompt>
First a PromptManager object is created, which takes a series of initial tags that you'd like to remain active over multiple prompts (I find it useful to add LoRA triggers here). Once the prompt manager is created a text box appears for the actual prompt. You can also add random tags pulled from resources/danbooru-tag.csv, shuffle your prompt tags, or generate a list of tags that you can manually add to the prompt.  

**ImageGenerator** handles methods for generation and upscaling, with a configuration collections.dataset that's controlled by widgets. 

### Note on Image2Image generation

Running a txt2ing and an img2img pipeline at the same time in colab quickly overloads even a1000 GPUs. To remedy this, I've made a version of the ImageGenerator class that can handle both txt2img and img2img generation with a StableDiffusionXLImg2Img pipeline alone, using pre-generated noise latents. This approach works decently, but it's not great. For the most part the biggest problem seems to be generating hands.
So, if you want to use img2img and hiRes, there are two lines of code in the notebook that need to be active. After cloning the repository, run:

```python
%cd /content/toolkit
!git checkout img2img
```

If you want better txt2img generation, comment these out and the notebook will run the main branch and load the basic StableDiffusionXLPipeline instead. I welcome any suggestions for a better approach!
