import os
from io import StringIO
import contextlib
from PIL import Image
import numpy as np
import torch
# from torchvision.tv_tensors import Image
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from download import download_file


def factorize(num: int, max_value: int) -> list[float]:
  result = []
  while num > max_value:
    result.append(max_value)
    num -= max_value
  result.append(round(num, 4))
  return result


def upscale(
    img_list: list[Image],
    model_name: str = "RealESRGAN_x4plus",
    scale_factor: float = 4,
    half_precision: bool = False,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0
) -> list[Image]:

    # Check Model
    if model_name == "RealESRGAN_x4plus":
        upscale_model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":
        upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    else:
        raise NotImplementedError("Model name not supported")

    # Download
    model_path = download_file(file_url,
                               path='./upscaler-model',
                               progress=False,
                               interrupt_check=False)

    # Declare model
    upsampler = RealESRGANer(
                            scale=netscale,
                            model_path=os.path.join('./upscaler-model', model_path),
                            model=upscale_model,
                            tile=tile,
                            tile_pad=tile_pad,
                            pre_pad=pre_pad,
                            half=half_precision,
                            gpu_id=None,
    )

    # Upscale
    torch.cuda.empty_cache()
    upscaled_images = []
    with tqdm(total=len(img_list)) as pb:
        for i, img in enumerate(img_list):
            image = np.array(img)
            outscale_list = factorize(scale_factor, netscale)
            with contextlib.redirect_stdout(StringIO()):
                for outscale in outscale_list:
                    current_image = upsampler.enhance(image, outscale=outscale)[0]
                    image = current_image
                upscaled_images.append(Image.fromarray(image))
            pb.update(1)

    return upscaled_images
