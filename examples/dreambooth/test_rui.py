from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch


# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("logs/checkpoint-1000/checkpoint-100/text_encoder")
from train_dreambooth import import_model_class_from_model_name_or_path
pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
revision = None
variant = None

text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
)
unet = UNet2DConditionModel.from_pretrained("logs/mimi_2/checkpoint-2000/unet")

pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path, unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")

# image = pipeline("A photo of sks cat looking shamefully at its pile of poop on the floor", num_inference_steps=50, guidance_scale=7.5).images[0]
image = pipeline("A photo of sks cat wearing a big cartoonish smile", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("mimi-bucket.png")