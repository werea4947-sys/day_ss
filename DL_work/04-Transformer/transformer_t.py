import argparse
import math
import os
import tarfile
import tempfile
from pathlib import Path

import warnings
# 抑制 PyTorch 关于 nested tensors 的原型阶段警告（无害，可忽略）
warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sacrebleu
import sentencepiece as spm

try:
    import swanlab
except Exception:
    swanlab = None
# 运行时是否成功初始化了 swanlab（启用且 init 成功才为 True）
swan_run = None

# SentencePiece 里固定特殊符号的 id，和下面的常量保持一致
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

DEFAULT_DATA = Path(__file__).with_name("实验4：数据集sample.tar.gz")
DEFAULT_WEIGHTS = Path(__file__).with_name("transformer_t_v2.pth")


# ----------------------------- 数据读取 -----------------------------
# 读取 tar.gz 里的文本文件，返回字符串列表；切分成对字符串；根据翻译方向交换源/目标
def read_member_lines(tar: tarfile.TarFile, member_name: str) -> list[str]:
    member = tar.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(member_name)
    return member.read().decode("utf-8", errors="ignore").splitlines()


def split_paired_lines(lines: list[str]) -> list[tuple[str, str]]:
    # 把交错排列的“源句/目标句”切成成对字符串（保留原始空格分词，交给 SP 再切子词）
    rows = [line.strip() for line in lines if line.strip()]
    pairs = []
    for i in range(0, len(rows) - 1, 2):
        if rows[i] and rows[i + 1]:
            pairs.append((rows[i], rows[i + 1]))
    return pairs


def load_from_tar(path: Path):
    # 读取训练集、开发集、测试集和参考译文，全部返回“字符串”而非已切分的 token 列表
    with tarfile.open(path, "r:gz") as tar:
        train_src = read_member_lines(tar, "sample-submission-version/TM-training-set/chinese.txt")
        train_tgt = read_member_lines(tar, "sample-submission-version/TM-training-set/english.txt")
        dev_lines = read_member_lines(tar, "sample-submission-version/Dev-set/Niu.dev.txt")
        test_lines = read_member_lines(tar, "sample-submission-version/Test-set/Niu.test.txt")
        ref_lines = read_member_lines(tar, "sample-submission-version/Reference-for-evaluation/Niu.test.reference")

    # 切分成对字符串，并根据翻译方向交换源/目标
    train_pairs = [(s.strip(), t.strip()) for s, t in zip(train_src, train_tgt) if s.strip() and t.strip()]
    dev_pairs = split_paired_lines(dev_lines)            # (zh, en)
    test_src = [line.strip() for line in test_lines if line.strip()]   # zh
    test_ref = [tgt for _, tgt in split_paired_lines(ref_lines)]       # en
    return train_pairs, dev_pairs, test_src, test_ref


def orient(pairs, direction):
    # zh2en：原始就是 (zh, en)；en2zh：交换为 (en, zh)
    if direction == "en2zh":
        return [(t, s) for s, t in pairs]
    return pairs


# ----------------------------- 子词词表 -----------------------------
def train_spm(texts: list[str], model_prefix: str, vocab_size: int, coverage: float):
    # 在训练语料上训练一个 SentencePiece(BPE) 模型，固定 pad/bos/eos/unk 的 id
    if Path(model_prefix + ".model").exists():
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(texts))
        tmp_path = f.name
    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=coverage,
            pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
            pad_piece="<pad>", bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>",
        )
    finally:
        os.unlink(tmp_path)


def load_spm(model_prefix: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")
    return sp


def sp_encode(sp, text: str) -> list[int]:
    return [BOS_ID] + sp.encode(text, out_type=int) + [EOS_ID]


def sp_decode(sp, ids: list[int]) -> str:
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(ids)


# ----------------------------- 数据集 -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, sp_src, sp_tgt, max_len):
        self.pairs = pairs
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        return (sp_encode(self.sp_src, src)[: self.max_len],
                sp_encode(self.sp_tgt, tgt)[: self.max_len])


def pad_sequences(batch, pad_idx):
    max_len = max(len(seq) for seq in batch)
    data = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch):
        data[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return data


def collate(batch):
    src_batch, tgt_batch = zip(*batch)
    return pad_sequences(list(src_batch), PAD_ID), pad_sequences(list(tgt_batch), PAD_ID)


# ----------------------------- 模型 -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model 
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID) #词表嵌入
        self.pos = PositionalEncoding(d_model, dropout) #位置嵌入
        self.transformer = nn.Transformer( #transformer主体
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_pad): #编码器输入：源句子和源句子padding掩码；输出：编码器记忆
        src_e = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_e, src_key_padding_mask=src_pad)

    def decode(self, tgt, memory, src_pad, tgt_pad): #解码器输入：目标句子、编码器记忆、源句子padding掩码、目标句子padding掩码；输出：解码器输出（未归一化的词表分布）
        tgt_e = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
        out = self.transformer.decoder(
            tgt_e, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.fc(out)

    def forward(self, src, tgt, src_pad, tgt_pad):
        memory = self.encode(src, src_pad)
        return self.decode(tgt, memory, src_pad, tgt_pad)


# ----------------------------- 学习率调度（Noam warmup） -----------------------------
def noam_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------- 训练与解码 -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_pad, tgt_pad = src.eq(PAD_ID), tgt_in.eq(PAD_ID)
        logits = model(src, tgt_in, src_pad, tgt_pad)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def batched_greedy_decode(model, src, device, max_len=80):
    # 一次编码、逐步解码的批量贪心，比逐句解码快很多
    model.eval()
    src = src.to(device)
    src_pad = src.eq(PAD_ID)
    memory = model.encode(src, src_pad)
    b = src.size(0)
    ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    for _ in range(max_len):
        tgt_pad = ys.eq(PAD_ID)
        logits = model.decode(ys, memory, src_pad, tgt_pad)
        nxt = logits[:, -1].argmax(-1)
        nxt = nxt.masked_fill(finished, PAD_ID)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        finished = finished | nxt.eq(EOS_ID)
        if bool(finished.all()):
            break
    return ys.tolist()


@torch.no_grad()
def evaluate(model, pairs, sp_src, sp_tgt, device, tgt_lang, max_len=80, batch_size=64):
    model.eval()
    candidates, references = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        src_ids = [sp_encode(sp_src, s)[:max_len] for s, _ in chunk]
        src = pad_sequences(src_ids, PAD_ID)
        outs = batched_greedy_decode(model, src, device, max_len)
        for (_, ref), out in zip(chunk, outs):
            candidates.append(sp_decode(sp_tgt, out))
            references.append(ref)
    if not candidates:
        return 0.0
    tok = "zh" if tgt_lang == "zh" else "13a"
    return float(sacrebleu.corpus_bleu(candidates, [references], tokenize=tok, force=True).score)


# ----------------------------- 主流程 -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Transformer MT (subword + warmup + label smoothing)")
    parser.add_argument("--mode", choices=["train", "eval", "translate"], default="train")
    parser.add_argument("--direction", choices=["zh2en", "en2zh"], default="zh2en")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--weights", type=str, default="", help="权重文件路径，若不指定则根据 --direction 使用 zh2en.pth/en2zh.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--patience", type=int, default=4, help="dev BLEU 多少轮不提升就早停")
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--no-early-stop", action="store_true", help="禁用基于 dev BLEU 的提前停止")
    # swanlab 日志（可选）：--use-swanlab 开启
    parser.add_argument("--use-swanlab", action="store_true", help="启用 swanlab 记录训练日志")
    parser.add_argument("--swanlab-project", type=str, default="transformer_mt")
    parser.add_argument("--swanlab-run-name", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt_lang = "en" if args.direction == "zh2en" else "zh"

    # 读取并按方向摆正
    train_pairs, dev_pairs, test_src, test_ref = load_from_tar(data_path)
    train_pairs = orient(train_pairs, args.direction)
    dev_pairs = orient(dev_pairs, args.direction)
    if args.direction == "en2zh":
        test_src, test_ref = test_ref, test_src   # 源变英文、参考变中文

    # 训练（或复用）两个 SentencePiece 子词模型；不同方向不同覆盖率
    # 如果用户没有传入显式的 --weights，则根据 --direction 使用方向相关的默认权重文件（zh2en.pth / en2zh.pth）
    if args.weights == str(DEFAULT_WEIGHTS) or not args.weights:
        w = Path(__file__).with_name(f"{args.direction}.pth")
    else:
        w = Path(args.weights)
    src_prefix = str(w.with_name(w.stem + f"_sp_src_{args.direction}"))
    tgt_prefix = str(w.with_name(w.stem + f"_sp_tgt_{args.direction}"))
    src_cov = 0.9995 if args.direction == "zh2en" else 1.0
    tgt_cov = 1.0 if args.direction == "zh2en" else 0.9995
    train_spm([s for s, _ in train_pairs], src_prefix, args.vocab_size, src_cov)
    train_spm([t for _, t in train_pairs], tgt_prefix, args.vocab_size, tgt_cov)
    sp_src, sp_tgt = load_spm(src_prefix), load_spm(tgt_prefix)

    model = TransformerMT(
        sp_src.get_piece_size(), sp_tgt.get_piece_size(),
        args.d_model, args.nhead, args.layers, args.ffn_dim, args.dropout).to(device)

    if args.mode == "train":
        global swan_run
        if args.use_swanlab:
            if swanlab is None:
                print("警告: 未安装 swanlab，跳过日志记录。安装: pip install swanlab")
            else:
                try:
                    swan_run = swanlab.init(
                        project=args.swanlab_project,
                        experiment_name=args.swanlab_run_name,
                        config=vars(args),
                    )
                except Exception as e:
                    print(f"swanlab 初始化失败，继续训练: {e}")
                    swan_run = None

        loader = DataLoader(
            TranslationDataset(train_pairs, sp_src, sp_tgt, args.max_len),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
        # Noam 配合 base lr=1.0，betas/eps 用原论文设置
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = noam_scheduler(optimizer, args.d_model, args.warmup)

        best_bleu, no_improve = 0.0, 0
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, scheduler, criterion, device)
            bleu = evaluate(model, dev_pairs, sp_src, sp_tgt, device, tgt_lang, args.max_len)
            print(f"epoch {epoch}/{args.epochs}  loss={loss:.4f}  dev_BLEU4={bleu:.2f}")
            if swan_run is not None:
                lr_now = scheduler.get_last_lr()[0]
                swanlab.log({"train/loss": loss, "dev/bleu4": bleu, "lr": lr_now}, step=epoch)
            if bleu > best_bleu:
                best_bleu, no_improve = bleu, 0
                torch.save({"model": model.state_dict(), "args": vars(args)}, w)
                print(f"  新最佳 BLEU={bleu:.2f}，已保存到 {w}")
            else:
                no_improve += 1
            # 根据用户选项决定是否提前停止
            if not args.no_early_stop:
                if bleu > 14:
                    print(f"dev BLEU4 已达到 {bleu:.2f} (>14)，提前停止。")
                    break
                if no_improve >= args.patience:
                    print(f"连续 {args.patience} 轮无提升，提前停止。")
                    break
        print(f"训练结束，最佳 dev BLEU4 = {best_bleu:.2f}")
        if swan_run is not None:
            swanlab.log({"dev/best_bleu4": best_bleu})
            try:
                swanlab.finish()
            except Exception:
                pass
        return

    # eval / translate 需要先加载权重
    if not w.exists():
        raise FileNotFoundError(f"未找到权重 {w}，请先用 --mode train 训练。")
    # 兼容不同保存格式：{"model": state_dict} 或直接 state_dict
    ckpt = torch.load(w, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model", ckpt.get("state_dict"))
    else:
        state = ckpt
    model.load_state_dict(state)

    if args.mode == "eval":
        dev_bleu = evaluate(model, dev_pairs, sp_src, sp_tgt, device, tgt_lang, args.max_len)
        print(f"dev  BLEU4: {dev_bleu:.2f}")
        if test_src and test_ref:
            n = min(len(test_src), len(test_ref))
            test_pairs = list(zip(test_src[:n], test_ref[:n]))
            test_bleu = evaluate(model, test_pairs, sp_src, sp_tgt, device, tgt_lang, args.max_len)
            print(f"test BLEU4: {test_bleu:.2f}")
        return

    if args.mode == "translate":
        if not args.src.strip():
            raise ValueError("translate 模式需要 --src")
        src = pad_sequences([sp_encode(sp_src, args.src.strip())[: args.max_len]], PAD_ID)
        out = batched_greedy_decode(model, src, device, args.max_len)[0]
        print(sp_decode(sp_tgt, out))


if __name__ == "__main__":
    main()