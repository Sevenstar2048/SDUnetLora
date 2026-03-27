from pathlib import Path

import torch
from diffusers import FluxPipeline

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_DIR = Path("./finetune/lora/flux_digimon")
OUTPUT_IMAGE = Path("./test_flux.png")


def pick_dtype_and_device() -> tuple[str, torch.dtype]:
    """根据硬件自动选择推理设备与精度。"""
    if torch.cuda.is_available():
        # Flux 在 CUDA 上通常用 bfloat16 更稳；若显卡不支持可改为 float16。
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def main() -> None:
    if not LORA_DIR.exists():
        raise FileNotFoundError(
            f"未找到 LoRA 目录: {LORA_DIR.as_posix()}。\n"
            "请先执行 trainFluxLoRA.py，或确认输出目录路径正确。"
        )

    device, dtype = pick_dtype_and_device()

    # 1) 先加载 Flux 基座模型。
    pipe = FluxPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)

    # 2) 再注入 LoRA 权重。优先加载根目录权重，若不存在则回退到最新 checkpoint。
    lora_weight = "pytorch_lora_weights.safetensors"
    if (LORA_DIR / lora_weight).exists():
        pipe.load_lora_weights(LORA_DIR.as_posix(), weight_name=lora_weight)
    else:
        checkpoints = sorted([p for p in LORA_DIR.glob("checkpoint-*") if p.is_dir()])
        if not checkpoints:
            raise FileNotFoundError(
                f"在 {LORA_DIR.as_posix()} 中未找到 LoRA 权重文件 {lora_weight}。"
            )
        latest_ckpt = checkpoints[-1]
        pipe.load_lora_weights(latest_ckpt.as_posix(), weight_name=lora_weight)

    pipe = pipe.to(device)

    # 3) 给定提示词做扩散采样，导出测试图。
    prompt = "a blue skin agumon, anime style, clean background, high detail"
    negative_prompt = "blurry, low quality, distorted"

    generator = torch.Generator(device=device).manual_seed(1024)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=256,
        generator=generator,
    ).images[0]

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_IMAGE)
    print(f"测试图已保存: {OUTPUT_IMAGE.as_posix()}")


if __name__ == "__main__":
    main()
