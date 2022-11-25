# Stable Edit Image and Command for stable diffusion



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
