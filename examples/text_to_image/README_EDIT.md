# Stable Edit Image and Command for stable diffusion

our data set is in the stable_edit folder.  two subfolders for both data types.  

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="stable_edit/reddit_data"
export TRAIN_DIR = "."

accelerate launch --mixed_precision="fp16"  train_text_and_image_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="stable_edit/output" 
```

I think we can get away with just using the inpainting script, it accepts an image, we are just adding it without any mask making
so it should look like this
```python
from diffusers import StableDiffusionImg2ImgPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]  ##This needs to be edited to take prompt and image
image.save("yoda-pokemon.png")
```
https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines#examples has the image to image example.