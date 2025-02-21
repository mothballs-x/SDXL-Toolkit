from matplotlib import pyplot as plt
from dataclasses import asdict, dataclass
from PIL import Image
from typing import Union, List
from math import ceil
from pathlib import Path
import time
import json

class ImageUtility:
  def __init__(self, root_dir, root_name="image", count=1):
    self.root_dir = Path(root_dir)
    self.root_name = root_name
    self.count = count
    self.log = Path(self.root_dir) / f'gen_log.json'
    if not self.log.exists():
        self.log.touch()
  
  def view_grid(self, images: List, scale_factor: float):
    num_images = len(images)
    grid_size = ceil(num_images ** 0.5)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            w, h = img.size
            scaled_img = img.resize((int(w // scale_factor), int(h // scale_factor)))
            ax.imshow(scaled_img)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

  def show_scaled(self, image, factor=2):
    # Resize the image
    new_size = (int(image.size[0] // factor), int(image.size[1] // factor))
    image = image.resize(new_size)
    
    # Display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()

  def log_config(self, img_path: str, generator_config: dataclass) -> None:
      if self.log.stat().st_size == 0:
          data = []
      else:
          with open(self.log, 'r') as json_in:
              try:
                  data = json.load(json_in)
                  if not isinstance(data, list):
                      data = []
              except json.JSONDecodeError:
                  data = []

      new_data = {'file': str(img_path)}
      new_data.update(asdict(generator_config))
      data.append(new_data)

      with open(self.log, 'w') as json_out:
          json.dump(data, json_out, indent=4)


  def save(self, image, logging=False, config=None):
    now = time.localtime()
    image_path = self.root_dir / f"{self.root_name}_{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}{now.tm_min}-{self.count}.png"
    if logging and config:
        self.log_config(image_path.name, config)
    image.save(image_path)
    self.count += 1
    return image_path

