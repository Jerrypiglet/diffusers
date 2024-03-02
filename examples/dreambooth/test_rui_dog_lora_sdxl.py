'''
trained following the setting in README:

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export INSTANCE_DIR="dog"

accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=logs/dog_lora_sdxl \
    --instance_prompt="a photo of sks dog" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 \
    --validation_prompt="A photo of sks dog in a bucket" \
    --validation_epochs=50 \
    --checkpointing_steps=200 \
    --seed 0 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing

'''
from diffusers import DiffusionPipeline
import torch

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("logs/checkpoint-1000/checkpoint-100/text_encoder")
from train_dreambooth import import_model_class_from_model_name_or_path
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
revision = None
variant = None

text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
)

pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, dtype=torch.float16,).to("cuda")

# load trained weights here (lora weights, only)
# pipeline.load_lora_weights("logs/dog_lora_v1-5/checkpoint-500")
pipeline.load_lora_weights("logs/dog_lora_v1-5_textencoder/checkpoint-500")

image = pipeline("A photo of sks dog in a bucket", num_inference_steps=25).images[0]
image.save("results/dog-bucket-lora.png")