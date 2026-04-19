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
SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-2
VAL_RATIO = 0.1
NUM_WORKERS = 2
IMAGE_SIZE = 32
NUM_CLASSES = 10
PATCH_SIZE = 4
DIM = 256
DEPTH = 6
HEADS = 8
DIM_HEAD = 64
MLP_DIM = 512
DROPOUT = 0.1
EMB_DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
SAVE_PATH = Path("vit_cifar10_best.pth")
DATA_ROOT = Path("./data")


# ============================== 1) Utility functions ==============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return 100.0 * (preds == labels).sum().item() / labels.size(0)


# ============================== 2) ViT modules ==============================
def pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
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
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        batch_size, num_tokens, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.view(batch_size, num_tokens, self.heads, self.dim_head).transpose(1, 2)
            for tensor in qkv
        ]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
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
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
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

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        assert pool in {"cls", "mean"}

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        self.pool = pool

        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.patch_proj_norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
        nn.init.zeros_(self.mlp_head.bias)

    def to_patch_embedding(self, img: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = img.shape
        patches = img.view(
            batch_size,
            channels,
            self.num_patches_h,
            self.patch_height,
            self.num_patches_w,
            self.patch_width,
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        patches = patches.view(batch_size, self.num_patches_h * self.num_patches_w, -1)
        patches = self.patch_norm(patches)
        patches = self.patch_proj(patches)
        return self.patch_proj_norm(patches)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : num_patches + 1]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


# ============================== 3) Data pipeline ==============================
def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    val_size = int(len(full_train) * VAL_RATIO)
    train_size = len(full_train) - val_size

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_set.dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=eval_transform)

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
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

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
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
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
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()

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

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}% | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to: {SAVE_PATH}")

    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss: {test_loss:.4f}, acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
