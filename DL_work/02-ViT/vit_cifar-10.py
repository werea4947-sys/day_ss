
# ============================== 0) 超参数区（统一放在最前） ==============================
# 随机种子：用于保证结果尽量可复现。
SEED = 42
# 训练批大小。
BATCH_SIZE = 128
# 训练轮数。
EPOCHS = 20
# 初始学习率。
LEARNING_RATE = 3e-4
# AdamW 的权重衰减系数。
WEIGHT_DECAY = 1e-4
# 从训练集里划分验证集的比例。
VAL_RATIO = 0.2
# DataLoader 的子进程数量。
NUM_WORKERS = 2
# 输入图像尺寸（变换与模型保持一致）。
IMAGE_SIZE = 32
# CIFAR-10 的类别数。
NUM_CLASSES = 10
# ViT 的 patch 尺寸（必须整除 IMAGE_SIZE）。
PATCH_SIZE = 4
# token 向量的嵌入维度。
EMBED_DIM = 192
# 多头注意力的头数。
NUM_HEADS = 3
# Transformer 编码器层数。
DEPTH = 6
# Transformer 块中 MLP 的扩展倍率。
MLP_RATIO = 4
# Dropout 概率。
DROPOUT = 0.1
# 交叉熵损失中的标签平滑系数。
LABEL_SMOOTHING = 0.0
# 最优模型权重保存路径。
SAVE_PATH = "vit_cifar10_best.pth"
# 数据集根目录。
DATA_ROOT = "./data"

# ============================== 1) 导入依赖 ==============================
# 标准库：设置随机种子。
import random
# 类型标注：让函数签名更清晰。
from typing import Tuple
# NumPy：用于随机种子控制。
import numpy as np
# PyTorch 核心库。
import torch
# 神经网络模块。
import torch.nn as nn
# 优化器模块。
import torch.optim as optim
# 数据加载工具。
from torch.utils.data import DataLoader, random_split
# Torchvision 数据集。
from torchvision import datasets
# Torchvision 图像变换。
from torchvision import transforms


# ============================== 2) 工具函数 ==============================
def set_seed(seed: int) -> None:
	"""设置随机种子，尽量保证可复现。"""
	# Python 随机种子。
	random.seed(seed)
	# NumPy 随机种子。
	np.random.seed(seed)
	# PyTorch CPU 随机种子。
	torch.manual_seed(seed)
	# PyTorch CUDA 随机种子（单卡）。
	torch.cuda.manual_seed(seed)
	# PyTorch CUDA 随机种子（多卡）。
	torch.cuda.manual_seed_all(seed)
	# 启用 CuDNN 确定性模式。
	torch.backends.cudnn.deterministic = True
	# 关闭 benchmark，避免自动算法选择带来的波动。
	torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
	"""选择运行设备：优先 GPU，否则使用 CPU。"""
	# 如果可用则选择 CUDA。
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 返回设备对象。
	return device


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
	"""计算 Top-1 准确率（百分比）。"""
	# 取每个样本预测概率最大的类别索引。
	preds = torch.argmax(logits, dim=1)
	# 统计预测正确的样本数。
	correct = (preds == labels).sum().item()
	# 计算百分比准确率。
	acc = 100.0 * correct / labels.size(0)
	# 返回标量准确率。
	return acc


# ============================== 3) ViT 模型组件 ==============================
class PatchEmbedding(nn.Module):
	"""把图像切成 patch，并将每个 patch 映射到嵌入向量。"""

	def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
		# 初始化父类。
		super().__init__()
		# 确保 patch_size 可以整除 img_size。
		assert img_size % patch_size == 0, "patch_size must divide img_size"
		# 每张图像的 patch 总数。
		self.num_patches = (img_size // patch_size) ** 2
		# 使用卷积同时完成“切块 + 线性投影”。
		self.proj = nn.Conv2d(
			in_channels=in_chans,
			out_channels=embed_dim,
			kernel_size=patch_size,
			stride=patch_size,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 输入 x 形状: [B, C, H, W]。
		# 卷积投影后: [B, embed_dim, H/ps, W/ps]。
		x = self.proj(x)
		# 展平空间维度: [B, embed_dim, num_patches]。
		x = x.flatten(2)
		# 交换维度得到 token 布局: [B, num_patches, embed_dim]。
		x = x.transpose(1, 2)
		# 返回 patch token 序列。
		return x

"""简化版 Vision Transformer"""
class SimpleViT(nn.Module):

	def __init__(
		self,
		img_size: int = 32,
		patch_size: int = 4,
		in_chans: int = 3,
		num_classes: int = 10,
		embed_dim: int = 192,
		depth: int = 6,
		num_heads: int = 3,
		mlp_ratio: int = 4,
		dropout: float = 0.1,
	) -> None:
		# 初始化父类。
		super().__init__()
		# 构建 patch embedding 层。
		self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
		# 记录 patch 数量。
		num_patches = self.patch_embed.num_patches

		# 可学习的分类 token（每个样本一个）。
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		# 可学习的位置编码（包含 cls token 和 patch token）。
		self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
		# 加位置编码后的 dropout。
		self.pos_drop = nn.Dropout(p=dropout)

		# 定义单个 Transformer 编码层。
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=embed_dim * mlp_ratio,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)
		# 堆叠多个编码层。
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

		# 最终层归一化。
		self.norm = nn.LayerNorm(embed_dim)
		# 分类头。
		self.head = nn.Linear(embed_dim, num_classes)

		# 初始化参数。
		self._init_weights()

	def _init_weights(self) -> None:
		"""初始化可学习参数。"""
		# cls token 使用截断正态分布初始化。
		nn.init.trunc_normal_(self.cls_token, std=0.02)
		# 位置编码使用截断正态分布初始化。
		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		# 遍历模块并初始化线性层和归一化层。
		for m in self.modules():
			# 线性层初始化。
			if isinstance(m, nn.Linear):
				# 权重使用 Xavier 均匀分布。
				nn.init.xavier_uniform_(m.weight)
				# 若存在偏置则置零。
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			# LayerNorm 初始化。
			elif isinstance(m, nn.LayerNorm):
				# 缩放参数置为 1。
				nn.init.ones_(m.weight)
				# 偏置置为 0。
				nn.init.zeros_(m.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 图像切块并投影: [B, N, D]。
		x = self.patch_embed(x)
		# 获取 batch 大小。
		bsz = x.size(0)
		# 扩展 cls token 到每个样本: [B, 1, D]。
		cls_tokens = self.cls_token.expand(bsz, -1, -1)
		# 拼接 cls token 与 patch token: [B, 1+N, D]。
		x = torch.cat((cls_tokens, x), dim=1)
		# 加位置编码。
		x = x + self.pos_embed
		# 应用 dropout。
		x = self.pos_drop(x)
		# 送入 Transformer 编码器。
		x = self.encoder(x)
		# 取 cls token 输出。
		cls_out = x[:, 0]
		# 对 cls token 做归一化。
		cls_out = self.norm(cls_out)
		# 线性层输出 logits。
		logits = self.head(cls_out)
		# 返回分类 logits。
		return logits


# ============================== 4) 数据流程 ==============================
def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
	"""构建 CIFAR-10 的训练/验证/测试 DataLoader。"""
	# 训练集变换：包含轻量数据增强。
	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(IMAGE_SIZE, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
		]
	)
	# 验证/测试变换：不做随机增强。
	eval_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
		]
	)

	# 加载完整训练集（使用训练变换）。
	full_train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
	# 计算验证集大小。
	val_size = int(len(full_train) * VAL_RATIO)
	# 计算训练子集大小。
	train_size = len(full_train) - val_size
	# 按固定随机种子划分训练/验证子集。
	train_set, val_set = random_split(
		full_train,
		lengths=[train_size, val_size],
		generator=torch.Generator().manual_seed(SEED),
	)

	# 验证子集改为评估变换，避免数据增强影响评估稳定性。
	val_set.dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=eval_transform)

	# 加载测试集。
	test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=eval_transform)

	# 构建训练 DataLoader。
	train_loader = DataLoader(
		train_set,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)
	# 构建验证 DataLoader。
	val_loader = DataLoader(
		val_set,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)
	# 构建测试 DataLoader。
	test_loader = DataLoader(
		test_set,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)

	# 返回三个 DataLoader。
	return train_loader, val_loader, test_loader


# ============================== 5) 训练与评估循环 ==============================
def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	device: torch.device,
) -> Tuple[float, float]:
	"""训练一个 epoch，返回平均损失与准确率。"""
	# 切换到训练模式。
	model.train()
	# 累积损失。
	total_loss = 0.0
	# 累积正确预测数。
	total_correct = 0
	# 累积样本总数。
	total_samples = 0

	# 遍历每个小批次。
	for images, labels in loader:
		# 图像移动到目标设备。
		images = images.to(device)
		# 标签移动到目标设备。
		labels = labels.to(device)

		# 清空上一轮梯度。
		optimizer.zero_grad()
		# 前向传播。
		logits = model(images)
		# 计算损失。
		loss = criterion(logits, labels)
		# 反向传播。
		loss.backward()
		# 参数更新。
		optimizer.step()

		# 按批大小加权累计损失。
		total_loss += loss.item() * labels.size(0)
		# 累计本批正确预测数。
		total_correct += (logits.argmax(dim=1) == labels).sum().item()
		# 累计本批样本数。
		total_samples += labels.size(0)

	# 计算 epoch 平均损失。
	avg_loss = total_loss / total_samples
	# 计算 epoch 平均准确率。
	avg_acc = 100.0 * total_correct / total_samples
	# 返回指标。
	return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> Tuple[float, float]:
	"""评估模型，返回平均损失与准确率。"""
	# 切换到评估模式。
	model.eval()
	# 累积损失。
	total_loss = 0.0
	# 累积正确预测数。
	total_correct = 0
	# 累积样本总数。
	total_samples = 0

	# 遍历每个小批次。
	for images, labels in loader:
		# 图像移动到目标设备。
		images = images.to(device)
		# 标签移动到目标设备。
		labels = labels.to(device)

		# 前向传播。
		logits = model(images)
		# 计算损失。
		loss = criterion(logits, labels)

		# 按批大小加权累计损失。
		total_loss += loss.item() * labels.size(0)
		# 累计本批正确预测数。
		total_correct += (logits.argmax(dim=1) == labels).sum().item()
		# 累计本批样本数。
		total_samples += labels.size(0)

	# 计算平均损失。
	avg_loss = total_loss / total_samples
	# 计算平均准确率。
	avg_acc = 100.0 * total_correct / total_samples
	# 返回指标。
	return avg_loss, avg_acc


# ============================== 6) 主流程（完整逻辑链） ==============================
def main() -> None:
	"""执行完整流程：环境设置 -> 数据 -> 模型 -> 训练 -> 验证 -> 测试。"""
	# 设置随机种子。
	set_seed(SEED)
	# 选择训练设备。
	device = get_device()
	# 打印当前设备。
	print(f"当前设备: {device}")

	# 构建数据加载器。
	train_loader, val_loader, test_loader = build_dataloaders()
	# 实例化模型。
	model = SimpleViT(
		img_size=IMAGE_SIZE,
		patch_size=PATCH_SIZE,
		in_chans=3,
		num_classes=NUM_CLASSES,
		embed_dim=EMBED_DIM,
		depth=DEPTH,
		num_heads=NUM_HEADS,
		mlp_ratio=MLP_RATIO,
		dropout=DROPOUT,
	).to(device)

	# 定义损失函数。
	criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
	# 定义优化器。
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	# 定义学习率调度器。
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

	# 记录最佳验证准确率。
	best_val_acc = 0.0

	# epoch 训练循环。
	for epoch in range(1, EPOCHS + 1):
		# 训练一个 epoch。
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		# 每轮结束后做验证。
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		# 更新学习率。
		scheduler.step()

		# 打印本轮训练与验证指标。
		print(
			f"Epoch [{epoch:02d}/{EPOCHS}] | "
			f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}% | "
			f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%"
		)

		# 根据验证准确率保存最优模型。
		if val_acc > best_val_acc:
			# 更新最佳分数。
			best_val_acc = val_acc
			# 保存权重。
			torch.save(model.state_dict(), SAVE_PATH)
			# 打印保存信息。
			print(f"  -> 已保存当前最优模型: {SAVE_PATH}")

	# 加载最优权重，进行最终测试。
	model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
	# 在测试集上评估。
	test_loss, test_acc = evaluate(model, test_loader, criterion, device)
	# 打印最终测试指标。
	print(f"测试集 | loss: {test_loss:.4f}, acc: {test_acc:.2f}%")


# ============================== 7) 程序入口 ==============================
if __name__ == "__main__":
	# 启动完整训练与评估流程。
	main()
