from pathlib import Path
from collections import OrderedDict
from dataclasses import asdict
import json


def log_generation(image_path,
                   log_path,
                   model_name,
                   generator,
                   prompt_manager,
                   lora_manager,
                   seed,
                  ):
  if not Path(log_path).exists():
    with open(log_path, 'w') as f:
      json.dump([], f)
  if Path(log_path).stat().st_size == 0:
    data = []
  else:
    with open(log_path, 'r') as f:
      try:
        data = json.load(f)
        if not isinstance(data, list):
          if isinstance(data, dict):
            data = [data]
          else:
            data = []
      except json.JSONDecodeError:
        data = []

    new_data = OrderedDict(
        {'file': Path(image_path).name, 'model': model_name}
        )
    new_data.update(asdict(generator.config))
    new_data.update({'seed': seed})

    # print(type(prompt_manager))
    if type(prompt_manager).__name__ == "PromptManager":
      new_data.update({
        "pos_prompt": prompt_manager.pos_prompt,
        "neg_prompt": prompt_manager.neg_prompt,
    })
    else:
      print('prompt_manager must be Prompter class')

    new_data.update({
        lora.name: lora.weight for lora in lora_manager.loras.values()
    })

    data.append(new_data)
  with open(log_path, 'w') as f:
    json.dump(data, f, indent=2)