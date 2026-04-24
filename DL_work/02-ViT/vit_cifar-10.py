import random
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================== 0) Hyperparameters ==============================
# 随机种子，保证实验结果尽量可复现
SEED = 42
# 每个 batch 的样本数
BATCH_SIZE = 128
# 训练轮数
EPOCHS = 20
# 初始学习率
LEARNING_RATE = 3e-4
# AdamW 的权重衰减系数
WEIGHT_DECAY = 5e-2
# 从训练集划分验证集的比例
VAL_RATIO = 0.2
# DataLoader 子进程数
NUM_WORKERS = 2
# CIFAR-10 图像尺寸
IMAGE_SIZE = 32
# 分类类别数
NUM_CLASSES = 10
# Patch 边长
PATCH_SIZE = 4
# token 嵌入维度
DIM = 256
# Transformer 编码块层数
DEPTH = 6
# 多头注意力头数
HEADS = 8
# 每个注意力头的维度
DIM_HEAD = 64
# 前馈网络隐藏层维度
MLP_DIM = 512
# Transformer 内部 dropout
DROPOUT = 0.1
# Patch embedding 后的 dropout
EMB_DROPOUT = 0.1
# 标签平滑系数
LABEL_SMOOTHING = 0.1
# 最优模型保存路径
SAVE_PATH = Path("vit_cifar10_best.pth")
# 数据集根目录
DATA_ROOT = Path("./data")


# ============================== 1) Utility functions ==============================
def set_seed(seed: int) -> None:
    # 设置 Python / NumPy / PyTorch 的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 固定 CuDNN 算法，减少结果波动
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    # 优先使用 GPU，否则退回 CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # 取每个样本预测概率最大的类别
    preds = logits.argmax(dim=1)
    return 100.0 * (preds == labels).sum().item() / labels.size(0)


# ============================== 2) ViT modules ==============================
def pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    # 将单个整数统一转成 (h, w) 二元组，便于同时处理高和宽
    if isinstance(value, tuple):
        return value
    return (value, value)


class FeedForward(nn.Module):
    """Transformer 中的前馈网络模块"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        # 结构：LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """多头自注意力模块"""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        # 多头拼接后的总维度
        inner_dim = dim_head * heads
        # 当只有一个头且维度相同时，输出投影层可省略
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        # 缩放因子，防止点积过大
        self.scale = dim_head ** -0.5

        # 先归一化，再做注意力计算
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # 一次线性映射同时生成 Q、K、V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x = self.norm(x)
        batch_size, num_tokens, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 变形成多头格式: [B, heads, N, dim_head]
        q, k, v = [
            tensor.view(batch_size, num_tokens, self.heads, self.dim_head).transpose(1, 2)
            for tensor in qkv
        ]

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))

        # 注意力加权求和，并还原回 [B, N, inner_dim]
        out = torch.matmul(attn, v) #matmul是矩阵乘法，点积注意力的核心计算
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    """由多层 Attention + FeedForward 堆叠而成的 Transformer 编码器"""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        # 每层包含一个注意力模块和一个前馈模块
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            # 残差连接
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    """Vision Transformer 主体结构"""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 图像尺寸必须能被 patch 尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        # 池化方式只能是 cls token 或 mean pooling
        assert pool in {"cls", "mean"}

        # patch 总数与每个 patch 拉平后的维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 记录 patch 划分信息，方便前向传播时切块
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        self.pool = pool

        # Patch embedding：先归一化，再线性映射到 token 维度
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.patch_proj_norm = nn.LayerNorm(dim)

        # 可学习的位置编码和分类 token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer 编码器与最终分类头
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # 对关键可学习参数做初始化
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
        nn.init.zeros_(self.mlp_head.bias)

    def to_patch_embedding(self, img: torch.Tensor) -> torch.Tensor:
        # 输入 img: [B, C, H, W]
        batch_size, channels, height, width = img.shape
        # 按 patch 重新整理张量形状
        patches = img.view(
            batch_size,
            channels,
            self.num_patches_h,
            self.patch_height,
            self.num_patches_w,
            self.patch_width,
        )
        # 调整维度顺序，得到 [B, num_patches_h, num_patches_w, ph, pw, C]
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        # 拉平成 [B, num_patches, patch_dim]
        patches = patches.view(batch_size, self.num_patches_h * self.num_patches_w, -1)
        patches = self.patch_norm(patches)
        patches = self.patch_proj(patches)
        return self.patch_proj_norm(patches)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 将图像转换为 patch token 序列
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        # 为每个样本扩展一份 cls token，并拼接到序列开头
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 加上位置编码
        x = x + self.pos_embedding[:, : num_patches + 1]
        x = self.dropout(x)

        # 输入 Transformer 编码器
        x = self.transformer(x)
        # 根据池化方式得到整张图像的全局表征
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        # 输出分类 logits
        return self.mlp_head(x)


# ============================== 3) Data pipeline ==============================
def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    # 训练集增强：随机裁剪、翻转和 AutoAugment
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # 验证/测试集不做随机增强
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # 先加载完整训练集，再手动划分训练集和验证集
    full_train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    val_size = int(len(full_train) * VAL_RATIO)
    train_size = len(full_train) - val_size

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    # 验证集改用无增强的 transform，避免评估结果波动
    val_set.dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=eval_transform)

    # GPU 训练时开启 pin_memory 提升数据搬运效率
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, test_loader


# ============================== 4) Train and eval ==============================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    # 训练模式
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 将数据搬到训练设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 标准训练流程：清梯度 -> 前向 -> 损失 -> 反向 -> 更新
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 累加整个 epoch 的损失与正确数
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    # 验证/测试模式
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 验证时只前向计算，不更新参数
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# ============================== 5) Main ==============================
def main() -> None:
    # 初始化随机种子和训练设备
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # 构建训练集、验证集和测试集的 DataLoader
    train_loader, val_loader, test_loader = build_dataloaders()

    # 实例化 ViT 模型
    model = ViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        pool="cls",
        channels=3,
        dim_head=DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
    ).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 记录历史最优验证集准确率
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # 先训练一轮，再在验证集上评估
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}% | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%"
        )

        # 若当前验证准确率更高，则保存最优模型参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to: {SAVE_PATH}")

    # 训练完成后加载最佳模型，并在测试集上评估最终结果
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss: {test_loss:.4f}, acc: {test_acc:.2f}%")


if __name__ == "__main__":
    # 程序入口
    main()
