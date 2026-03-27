import json
import os
import random
import shutil
import subprocess
from pathlib import Path

# 训练脚本路径：通常来自 diffusers 官方 examples 的 train_dreambooth_lora_flux.py。
# 你可以把该脚本放在项目根目录，或改成你的实际路径。
TRAIN_SCRIPT = "./train_dreambooth_lora_flux.py"

# Flux 基座模型（需要先在 Hugging Face 上申请/同意对应模型许可）。
BASE_MODEL = "black-forest-labs/FLUX.1-dev"

# 原始数据目录（包含图片 + metadata.jsonl）
SOURCE_TRAIN_DIR = Path("./train")
SOURCE_METADATA = SOURCE_TRAIN_DIR / "metadata.jsonl"

# 自动生成的子集目录：用于只训练部分样本，降低显存/时间成本。
SUBSET_TRAIN_DIR = Path("./train_subset_100")
MAX_TRAIN_SAMPLES = 100

# Flux DreamBooth 脚本要求 instance_data_dir 为纯图片目录。
INSTANCE_IMAGE_DIR = Path("./train_flux_instance_images")

# LoRA 权重和检查点输出目录。
OUTPUT_DIR = Path("./finetune/lora/flux_digimon")

# 可选：从某个检查点继续训练。
# - 设为 None: 每次从头开始训练
# - 设为 "latest": 自动从 output_dir 中最新检查点继续
# - 设为 "checkpoint-500": 从指定检查点继续
RESUME_FROM_CHECKPOINT = None

# 省显存策略：默认关闭训练中验证（验证阶段最容易爆显存）。
ENABLE_VALIDATION_DURING_TRAINING = False


def prepare_subset_dataset(
    source_dir: Path,
    source_meta: Path,
    target_dir: Path,
    max_samples: int,
    seed: int,
) -> int:
    """从完整训练集随机采样 max_samples 条，复制图片并重建 metadata.jsonl。"""
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    rng = random.Random(seed)

    if not source_meta.exists():
        raise FileNotFoundError(
            f"未找到标注文件: {source_meta.as_posix()}。\n"
            "本脚本按【图片+文本】训练，需要 metadata.jsonl。"
        )

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with source_meta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            file_name = item.get("file_name")
            if not file_name:
                continue
            image_path = source_dir / file_name
            if image_path.exists() and image_path.suffix.lower() in image_suffixes:
                records.append(item)

    if not records:
        raise RuntimeError(
            f"{source_meta.as_posix()} 中没有可用样本，请检查 file_name 与图片是否匹配。"
        )

    rng.shuffle(records)
    subset = records[: min(max_samples, len(records))]

    with (target_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for item in subset:
            file_name = item["file_name"]
            shutil.copy2(source_dir / file_name, target_dir / file_name)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已准备训练子集: {len(subset)} 张 -> {target_dir.as_posix()}")
    return len(subset)


def prepare_instance_image_dir(source_dir: Path, target_dir: Path) -> int:
    """从子集目录筛出图片，生成 Flux DreamBooth 可读取的纯图片目录。"""
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_paths = sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_suffixes
        ]
    )

    if not image_paths:
        raise RuntimeError(
            f"在 {source_dir.as_posix()} 中未找到可训练图片。"
            "请确认图片放在该目录，并使用常见后缀（png/jpg/jpeg/webp/bmp）。"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for old_file in target_dir.iterdir():
        if old_file.is_file() or old_file.is_symlink():
            old_file.unlink()

    # 使用软链接而非复制，节省磁盘空间。
    for src in image_paths:
        dst = target_dir / src.name
        dst.symlink_to(src.resolve())

    print(f"已准备 Flux 纯图片目录: {len(image_paths)} 张 -> {target_dir.as_posix()}")
    return len(image_paths)


def main() -> None:
    if not Path(TRAIN_SCRIPT).exists():
        raise FileNotFoundError(
            f"未找到训练脚本: {TRAIN_SCRIPT}\n"
            "请确认 train_dreambooth_lora_flux.py 在项目根目录。"
        )

    if not SOURCE_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"未找到训练数据目录: {SOURCE_TRAIN_DIR.as_posix()}。\n"
            "请先准备好图片数据（建议包含 metadata.jsonl）。"
        )

    seed = 1024

    # 第 1 步：准备训练数据子集（100 条）。
    # 这样可以先快速验证 Flux LoRA 训练流程，再决定是否扩大数据量。
    prepare_subset_dataset(
        source_dir=SOURCE_TRAIN_DIR,
        source_meta=SOURCE_METADATA,
        target_dir=SUBSET_TRAIN_DIR,
        max_samples=MAX_TRAIN_SAMPLES,
        seed=seed,
    )

    # 第 2 步：准备 Flux DreamBooth 纯图片目录。
    # 注意：train_dreambooth_lora_flux.py 的 --instance_data_dir 不能包含 metadata.jsonl。
    prepare_instance_image_dir(SUBSET_TRAIN_DIR, INSTANCE_IMAGE_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 第 3 步：组织训练命令并启动。
    # accelerate 的作用：统一管理单卡/多卡/混合精度启动方式。
    # 这里使用了一套“先稳定跑通”的 Flux LoRA 参数：优先避免 OOM，再逐步提质量。
    cmd = [
        "accelerate",
        "launch",
        "--num_processes=1",  # 强制单进程，避免多进程带来的额外显存占用与调试复杂度。
        TRAIN_SCRIPT,
        f"--pretrained_model_name_or_path={BASE_MODEL}",  # Flux 基座模型。
        f"--instance_data_dir={INSTANCE_IMAGE_DIR.as_posix()}",  # 训练图片目录（必须是纯图片）。
        "--instance_prompt=a digimon creature",  # 统一实例提示词（DreamBooth 的触发词语义）。
        "--resolution=512",  # 训练分辨率；更高会明显增加显存消耗。
        "--center_crop",  # 先缩放再中心裁剪，保证输入尺寸统一。
        "--random_flip",  # 随机左右翻转，增加数据多样性。
        "--train_batch_size=1",  # 每步喂 1 张图，最显存友好。
        "--gradient_accumulation_steps=4",  # 累积 4 步梯度再更新，等效放大 batch。
        "--gradient_checkpointing",  # 用算力换显存，降低反向传播峰值显存。
        "--cache_latents",  # 缓存 VAE latent，减少重复编码开销并降低训练波动。
        "--max_sequence_length=256",  # 文本最大长度，缩短可降低文本侧显存占用。
        "--max_train_steps=1000",  # 总优化步数，决定训练时长与拟合程度。
        "--learning_rate=1e-4",  # 学习率；过大易不稳定，过小收敛慢。
        "--max_grad_norm=1.0",  # 梯度裁剪，防止梯度爆炸。
        "--lr_scheduler=cosine",  # 学习率调度策略：余弦衰减。
        "--lr_warmup_steps=0",  # 学习率预热步数；0 表示不开启预热。
        "--rank=16",  # LoRA 秩；越大可学习能力越强，也更耗显存。
        "--lora_alpha=16",  # LoRA 缩放系数；常与 rank 设为相同量级。
        "--checkpointing_steps=500",  # 每 500 步保存一次断点。
        "--mixed_precision=fp16",  # 半精度训练，显著降低显存占用。
        f"--output_dir={OUTPUT_DIR.as_posix()}",  # LoRA 权重与日志输出目录。
        f"--seed={seed}",  # 固定随机种子，便于复现实验结果。
    ]

    if ENABLE_VALIDATION_DURING_TRAINING:
        # 验证会触发额外推理与 VAE 解码，显存峰值会升高；默认关闭更稳。
        cmd.extend(
            [
                "--validation_prompt=a blue skin agumon, anime style",
                "--num_validation_images=1",  # 每次验证生成 1 张图，控制显存与时间。
                "--validation_epochs=20",  # 每 20 个 epoch 验证一次，降低验证频率。
            ]
        )

    if RESUME_FROM_CHECKPOINT:
        cmd.append(f"--resume_from_checkpoint={RESUME_FROM_CHECKPOINT}")

    print("即将执行 Flux LoRA 训练命令:")
    print(" ".join(cmd))
    try:
        env = os.environ.copy()
        # 减少显存碎片导致的 OOM。
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("\n训练启动失败，常见原因如下：")
        print("1) 没有 Flux 模型权限（未登录 HF 或未同意模型许可）")
        print("2) 磁盘空间不足（下载模型时经常触发 No space left on device）")
        print("3) 显存不足（CUDA out of memory）")
        print("\n快速自检建议：")
        print("- 权限: 用 huggingface_hub 的 whoami/model_info 检查登录与访问")
        print("- 磁盘: 执行 df -h，重点看根分区 / 的 Avail 是否为 0")
        print("- 显存: 降低分辨率、batch_size 或 LoRA rank")
        raise e


if __name__ == "__main__":
    main()
