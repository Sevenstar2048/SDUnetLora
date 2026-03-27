from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image


# =========================
# 一、基础配置（新手先只改这里）
# =========================

# 基座模型：和你当前 SD LoRA 实验保持一致，便于对比。
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# ControlNet 模型：这里使用最常见的 Canny 边缘版本。
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

# 输入图：你可以换成任意参考图片。
INPUT_IMAGE = Path("./test.png")

# 输出目录。
OUTPUT_DIR = Path("./controlnet_outputs")

# 生成参数。
PROMPT = "a blue skin agumon, anime style, clean lineart, high detail" #想画什么
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark" #不想要什么
NUM_INFERENCE_STEPS = 30 #生成步数，越多细节越丰富，但也越慢。30-50 是常见范围。
GUIDANCE_SCALE = 7.0 #引导尺度，控制生成结果与提示词的相关性。一般 7-8 是不错的起点；过高可能过拟合提示词，过低可能不够精准。
CONTROLNET_CONDITIONING_SCALE = 0.9 #ControlNet 条件强度，范围通常在 0-2 之间。0 表示完全不使用 ControlNet 条件，1 表示完全使用，超过 1 则是放大 ControlNet 的影响。根据输入图和需求调整，建议从 0.5-1.0 开始尝试。
SEED = 1024 #随机种子，固定后每次生成结果一致，便于调参对比。你可以换成任意整数。


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    """优先使用 CUDA；无显卡时回退到 CPU。"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def build_canny_control_image(image: Image.Image) -> Image.Image:
    """把普通图片转换为 Canny 边缘图，供 ControlNet 作为结构条件输入。"""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "未安装 opencv-python。请先执行: pip install opencv-python"
        ) from e

    image_np = np.array(image.convert("RGB"))
    edges = cv2.Canny(image_np, 100, 200)
    edges_3c = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_3c)


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"未找到输入图: {INPUT_IMAGE.as_posix()}。"
            "请放一张参考图到该路径，或修改 INPUT_IMAGE。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device, dtype = pick_device_and_dtype()

    # 第 1 步：读取参考图，并生成 ControlNet 结构图（边缘图）。
    image = Image.open(INPUT_IMAGE).convert("RGB")
    control_image = build_canny_control_image(image)
    control_image_path = OUTPUT_DIR / "control_canny.png"
    control_image.save(control_image_path)

    # 第 2 步：加载 ControlNet + SD 基座模型。
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # 减少显存峰值，适合远程环境。
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(SEED)

    # 第 3 步：输入提示词 + 控制图，执行生成。
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        generator=generator,
    )
    output = result.images[0]

    output_path = OUTPUT_DIR / "controlnet_result.png"
    output.save(output_path)

    print("ControlNet 测试完成")
    print(f"输入图: {INPUT_IMAGE.as_posix()}")
    print(f"控制图: {control_image_path.as_posix()}")
    print(f"结果图: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
