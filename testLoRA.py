import torch
from PIL.Image import Image as PILImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
	StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
	DPMSolverMultistepScheduler,
)
from typing import cast

model_path = "runwayml/stable-diffusion-v1-5"
LoRA_path = "./finetune/lora/digimon"

# 推理时优先用 CUDA；若无显卡则自动回退到 CPU。
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 加载基础模型。LoRA 不会替换整个模型，而是给注意力层加一组低秩增量参数。
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)

# DPM-Solver 采样器通常比默认采样器收敛更快，推理质量/速度平衡更好。
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 把训练好的 LoRA 权重注入到 UNet 的注意力模块中。
pipe.unet.load_attn_procs(LoRA_path)
pipe.to(device)

# 文本提示词 -> 扩散采样 -> 得到图像。
# 这里使用 return_dict=False，返回 tuple，便于类型检查通过。
result = pipe(
	"a blue skin agumon",
	num_inference_steps=50,
	output_type="pil",
	return_dict=False,
)
images = cast(list[PILImage], result[0])
image = images[0]
image.save("test.png")