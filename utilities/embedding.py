# Embedding Helper Function
from safetensors.torch import load_file

def create_embedding(pipeline, path, token):
  embed = load_file(path)
  pipeline.load_textual_inversion(
    embed['clip_g'],
    token=token,
    text_encoder=pipeline.text_encoder_2,
    tokenizer=pipeline.tokenizer_2,
    mean_resizing=False
  )
  pipeline.load_textual_inversion(
    embed['clip_l'],
    token=token,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    mean_resizing=False
  )

  return token