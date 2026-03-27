import json
import random
import shutil
import subprocess
from pathlib import Path

# 训练脚本路径：通常来自 diffusers 官方 examples 的 train_text_to_image_lora.py。
# 你可以把该脚本放在项目根目录，或改成你的实际路径。
TRAIN_SCRIPT = "./train_text_to_image_lora.py"

# 原始数据目录（包含图片 + metadata.jsonl）
SOURCE_TRAIN_DIR = Path("./train")
# 自动生成的子集目录：用于只训练部分样本，降低显存/时间成本。
SUBSET_TRAIN_DIR = Path("./train_subset_100")
MAX_TRAIN_SAMPLES = 100

# 可选：从某个检查点继续训练。
# - 设为 None: 每次从头开始训练
# - 设为 "latest": 自动从 output_dir 中最新检查点继续
# - 设为 "checkpoint-500": 从指定检查点继续
RESUME_FROM_CHECKPOINT = None


def prepare_subset_dataset(
	source_dir: Path,
	target_dir: Path,
	max_samples: int,
	seed: int,
) -> None:
	"""从完整训练集随机采样 max_samples 条，复制图片并重建 metadata.jsonl。"""
	source_meta = source_dir / "metadata.jsonl"
	if not source_meta.exists():
		raise FileNotFoundError(f"未找到元数据文件: {source_meta}")

	if target_dir.exists():
		shutil.rmtree(target_dir)
	target_dir.mkdir(parents=True, exist_ok=True)
	target_meta = target_dir / "metadata.jsonl"

	records = []
	with source_meta.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			item = json.loads(line)
			image_name = item.get("file_name")
			if not image_name:
				continue
			image_path = source_dir / image_name
			if image_path.exists():
				records.append(item)

	if not records:
		raise RuntimeError("metadata.jsonl 中没有可用样本，请检查 train 目录。")

	# 固定随机种子，保证每次抽样可复现（便于对比实验结果）。
	rng = random.Random(seed)
	rng.shuffle(records)
	subset = records[: min(max_samples, len(records))]

	with target_meta.open("w", encoding="utf-8") as f:
		for item in subset:
			image_name = item["file_name"]
			src_image = source_dir / image_name
			dst_image = target_dir / image_name
			shutil.copy2(src_image, dst_image)
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

	print(f"已准备训练子集: {len(subset)} 张 -> {target_dir}")


def main() -> None:
	if not Path(TRAIN_SCRIPT).exists():
		raise FileNotFoundError(
			f"未找到训练脚本: {TRAIN_SCRIPT}\n"
			"请先下载 diffusers/examples/text_to_image/train_text_to_image_lora.py "
			"到项目目录，或修改 TRAIN_SCRIPT 为你的实际路径。"
		)

	seed = 1024
	# 第 1 步：准备训练数据。
	# 这里会从 train 中随机抽样 100 张，生成一个更小的数据子集，
	# 这样你可以先快速验证流程，再决定是否扩大数据量和训练步数。
	prepare_subset_dataset(
		source_dir=SOURCE_TRAIN_DIR,
		target_dir=SUBSET_TRAIN_DIR,
		max_samples=MAX_TRAIN_SAMPLES,
		seed=seed,
	)

	# 第 2 步：组织训练命令并启动。
	# accelerate 的作用：统一管理单卡/多卡/混合精度启动方式。
	# 这里使用 fp16 来降低显存占用，适合入门训练 LoRA。
	cmd = [
		"accelerate",
		"launch",
		TRAIN_SCRIPT,
		"--mixed_precision=fp16",  # 混合精度训练，降低显存占用并提升速度。
		"--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",  # 基座模型。
		f"--train_data_dir={SUBSET_TRAIN_DIR.as_posix()}",  # 训练数据目录（图片 + metadata.jsonl）。
		"--dataloader_num_workers=0",  # 读图进程数；Windows/新手先用 0 更稳。
		"--resolution=512",  # 训练分辨率；越高越吃显存。
		"--center_crop",  # 中心裁剪到正方形。
		"--random_flip",  # 随机左右翻转，增强数据多样性。
		"--train_batch_size=1",  # 每步喂入 1 张图（显存友好）。
		"--gradient_accumulation_steps=4",  # 累积 4 步梯度后再更新一次参数。
		"--max_train_steps=1000",  # 总优化步数，控制训练时长。
		"--learning_rate=1e-4",  # 学习率；过大易崩，过小收敛慢。
		"--max_grad_norm=1",  # 梯度裁剪，防止梯度爆炸。
		"--lr_scheduler=cosine",  # 学习率调度策略：余弦衰减。
		"--lr_warmup_steps=0",  # 预热步数；0 表示不开启 warmup。
		"--output_dir=./finetune/lora/digimon",  # LoRA 权重和检查点输出目录。
		"--checkpointing_steps=500",  # 每 500 步保存一次检查点。
		"--validation_prompt=a blue skin agumon",  # 训练中抽样验证用提示词。
		f"--seed={seed}",
	]

	if RESUME_FROM_CHECKPOINT:
		cmd.append(f"--resume_from_checkpoint={RESUME_FROM_CHECKPOINT}")

	# 第 3 步：执行训练。
	# 训练内部核心流程（简化版）：
	# 1) 文本提示词 -> 文本编码器得到条件向量
	# 2) 图片 -> VAE 编码到潜空间 latent
	# 3) 给 latent 加噪声，UNet 预测噪声
	# 4) 用预测噪声与真实噪声计算损失并反向传播
	# 5) 只更新 LoRA 小矩阵参数，基座大模型参数基本冻结
	# 6) 到达 checkpointing_steps 时保存一次权重
	print("即将执行训练命令:")
	print(" ".join(cmd))
	subprocess.run(cmd, check=True)


if __name__ == "__main__":
	main()