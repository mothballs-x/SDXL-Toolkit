from pathlib import Path
from compel import Compel, ReturnedEmbeddingsType

# Lora Manager Class
class LoraManager:

    def __init__(self, pipeline, lora_paths: List[Tuple[str, str]], civitai_token, lora_dir: str = "/content/loras"):
        self.pipeline = pipeline
        self.lora_dir = Path(lora_dir)  # Convert to Path object
        self.lora_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.civitai_token = civitai_token

        # Check missing paths
        paths = [path for path, _ in lora_paths if path.startswith('http') or Path(path).exists() ]
        if not paths:
            raise ValueError(f"Paths do not exist")

        # Initialize LoRAs
        self.loras = {}
        for path, name in lora_paths:
            self.add_lora(path, name)

    class Lora:
        def __init__(self, container, path: str, title: str, name: str, weight: float = 0.0):
            self.container = container
            self.path = path
            self.title = title
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
                wieght_name=self.title,
                adapter_name=self.name
                )

    def update_weights(self):
        """Apply updated weights to LoRAs in the pipeline"""
        lora_names = [lora.name for lora in self.loras.values()]
        lora_weights = [lora.weight for lora in self.loras.values()]
        self.pipeline.set_adapters(lora_names, lora_weights)


    def add_lora(self, path: str, name: str, weight: float = 0.0):
        """Add a LoRA either from a local file or from CivitAI"""
        if path.startswith("http"):
            model_number = path.split('/')[-1].split('?')[0]
            print(f'Civitai Model Number for {name}: {model_number}')
            path = self.get_lora_from_link(model_number, name)  # Download LoRA from link
        else:
            path = Path(path)

        if path.exists():
            self.loras[name] = LoraManager.Lora(self, path, path.name, name, weight)
            self.update_weights()
        else:
            print(f"[Warning] LoRA path '{path}' does not exist.")

    def delete_lora(self, names: Union[str, list]):
        """Remove a LoRA from the manager"""
        if not (isinstance(names, list) or isinstance(names, str)):
          raise TypeError("Names must be a string or a list of strings")
        if isinstance(names, str):
          names = [names]
        for name in names:
          if name in self.loras:
            del self.loras[name]
            self.update_weights()
          else:
            print(f"[Warning] LoRA '{name}' not found.")
        self.pipeline.delete_adapters(names)

    def list_loras(self):
        """Return all available LoRAs"""
        return {lora.name: lora.title for lora in self.loras.values()}

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
            print(f"Warning: Downloaded file does not have a .safetensors extension: {lora_path}")

        return lora_path

# Prompter Manager Class

class Prompter:
    pony_pos = 'score_9, score_8_up, score_7_up, score_6_up, rating_explicit'
    pony_neg = 'score_5, score_4, score_3, score_2, rating_safe'

    def __init__(
            self, pipeline,
            initial_tags=None,
            pos_tokens=None, neg_tokens=None,
            df=None, pony=True
            ):
        self.initial = intital_tags
        pos_tokens = ', '.join(pos_tokens) if isinstance(pos_tokens, list) else pos_tokens
        neg_tokens = ', '.join(neg_tokens) is isinstance(neg_tokens, list) else neg_tokens
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.pony = pony

        self.prompt = ''
        self.pos_prompt = ''
        self.neg_prompt = ''

        self.df = df

        self.compel = Compel(
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, False]
                )
            
