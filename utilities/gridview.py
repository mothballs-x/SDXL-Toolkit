import matplotlib.pyplot as plt
from math import ceil
from typing import List
from PIL import Image

class ImageGrid:
    def view_grid(self, images: List[Image.Image], scale_factor: float):
        num_images = len(images)
        grid_size = ceil(num_images ** 0.5)

        # Dynamically adjust figure size based on scale_factor
        fig, axes = plt.subplots(
            grid_size, grid_size, 
            figsize=(grid_size * 4 / scale_factor, grid_size * 4 / scale_factor)  # Adjust figure size
        )
        axes = axes.flatten() if num_images > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < num_images:
                img = images[i]
                w, h = img.size
                # Use normal division to avoid rounding to zero
                scaled_img = img.resize((int(w / scale_factor), int(h / scale_factor)))
                ax.imshow(scaled_img)
                ax.axis("off")
            else:
                ax.axis("off")  # Hide empty subplots

        plt.tight_layout()
        plt.show()

    def show_scaled(self, image: Image.Image, factor=2):
        # Resize the image
        new_size = (int(image.size[0] / factor), int(image.size[1] / factor))
        scaled_image = image.resize(new_size)

        # Dynamically adjust figure size to reflect scale
        fig = plt.figure(figsize=(scaled_image.width / 50, scaled_image.height / 50))
        plt.imshow(scaled_image)
        plt.axis("off")  # Hide axes
        plt.show()
