import argparse
import math
import tarfile 
from collections import Counter 
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sacrebleu


# 特殊符号：用于填充、句首、句尾和未登录词
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = SPECIALS
# 默认数据包路径：和脚本放在同一目录下
DEFAULT_DATA = Path(__file__).with_name("实验4：数据集sample.tar.gz")
# 默认权重保存路径：训练后写入同目录
DEFAULT_WEIGHTS = Path(__file__).with_name("transformer_t.pth")


def read_text_lines(path: Path) -> list[str]:
	# 读取普通文本文件；如果传入的是压缩包则不在这里处理
	if path.is_dir():
		return path.read_text(encoding="utf-8").splitlines()
	if path.suffixes[-2:] == [".tar", ".gz"]:
		raise ValueError("请使用 load_from_tar 读取 .tar.gz 数据包")
	return path.read_text(encoding="utf-8").splitlines()


def read_member_lines(tar: tarfile.TarFile, member_name: str) -> list[str]:
	# 从 tar.gz 数据包中读取指定成员文件的所有行
	member = tar.extractfile(member_name)
	if member is None:
		raise FileNotFoundError(member_name)
	return member.read().decode("utf-8", errors="ignore").splitlines()


def split_paired_lines(lines: list[str]) -> list[tuple[list[str], list[str]]]:
	# 将交错排列的“源句/目标句”行切成成对样本
	rows = [line.strip() for line in lines if line.strip()]
	if len(rows) < 2:
		return []
	pairs = []
	for i in range(0, len(rows) - 1, 2):
		src = rows[i].split()
		tgt = rows[i + 1].split()
		if src and tgt:
			pairs.append((src, tgt))
	return pairs

 
def load_from_tar(path: Path):
	# 读取实验数据包中的训练集、开发集、测试集和参考译文
	with tarfile.open(path, "r:gz") as tar:
		train_src = read_member_lines(tar, "sample-submission-version/TM-training-set/chinese.txt")
		train_tgt = read_member_lines(tar, "sample-submission-version/TM-training-set/english.txt")
		dev_lines = read_member_lines(tar, "sample-submission-version/Dev-set/Niu.dev.txt")
		test_lines = read_member_lines(tar, "sample-submission-version/Test-set/Niu.test.txt")
		ref_lines = read_member_lines(tar, "sample-submission-version/Reference-for-evaluation/Niu.test.reference")

	train_pairs = [(s.split(), t.split()) for s, t in zip(train_src, train_tgt) if s.strip() and t.strip()]
	dev_pairs = split_paired_lines(dev_lines)
	test_src = [line.split() for line in test_lines if line.strip()]
	test_ref_pairs = split_paired_lines(ref_lines)
	test_ref = [tgt for _, tgt in test_ref_pairs]
	return train_pairs, dev_pairs, test_src, test_ref


def build_vocab(seqs: list[list[str]]) -> dict[str, int]:
	# 按词频构建词表，并把特殊符号放在最前面
	counter = Counter(token for seq in seqs for token in seq)
	vocab = {token: idx for idx, token in enumerate(SPECIALS)}
	for token, _ in counter.most_common():
		if token not in vocab:
			vocab[token] = len(vocab)
	return vocab


def encode(seq: list[str], vocab: dict[str, int], add_bos: bool = False, add_eos: bool = False) -> list[int]:
	# 把词序列转成下标序列，可选加入句首和句尾标记
	ids = []
	if add_bos:
		ids.append(vocab[BOS])
	ids.extend(vocab.get(token, vocab[UNK]) for token in seq)
	if add_eos:
		ids.append(vocab[EOS])
	return ids


def pad_sequences(batch: list[list[int]], pad_idx: int) -> torch.Tensor:
	# 把不等长序列补齐成同一长度的张量
	max_len = max(len(seq) for seq in batch)
	data = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
	for i, seq in enumerate(batch):
		data[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
	return data


class TranslationDataset(Dataset):
	# 翻译数据集：每条样本返回“源句 + 目标句”
	def __init__(self, pairs: list[tuple[list[str], list[str]]], src_vocab: dict[str, int], tgt_vocab: dict[str, int]):
		self.pairs = pairs
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab

	def __len__(self) -> int:
		# 数据集样本数
		return len(self.pairs)

	def __getitem__(self, index: int):
		# 取出一条样本并编码
		src, tgt = self.pairs[index]
		return encode(src, self.src_vocab, add_bos=True, add_eos=True), encode(tgt, self.tgt_vocab, add_bos=True, add_eos=True)


def make_collate_fn(pad_idx_src: int, pad_idx_tgt: int):
	# DataLoader 的批处理函数：分别对源句和目标句补齐
	def collate(batch):
		src_batch, tgt_batch = zip(*batch)
		return pad_sequences(list(src_batch), pad_idx_src), pad_sequences(list(tgt_batch), pad_idx_tgt)

	return collate


class PositionalEncoding(nn.Module):
	# 正弦位置编码，为 Transformer 提供位置信息
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		# 预先计算所有位置的位置编码
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe.unsqueeze(0))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 把位置编码加到词向量上，再做 dropout
		x = x + self.pe[:, : x.size(1)]
		return self.dropout(x)


class TransformerMT(nn.Module):
	# 简化版 Transformer 翻译模型
	def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
		super().__init__()
		# 源语言和目标语言分别使用独立嵌入层
		self.src_embed = nn.Embedding(src_vocab_size, d_model)
		self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
		self.pos = PositionalEncoding(d_model, dropout)
		# 直接使用 PyTorch 自带 Transformer 组件
		self.transformer = nn.Transformer(
			d_model=d_model,
			nhead=nhead,
			num_encoder_layers=num_layers,
			num_decoder_layers=num_layers,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		# 最后一层线性映射到目标词表大小
		self.fc = nn.Linear(d_model, tgt_vocab_size)

	def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
		# 前向传播：源句编码、目标句解码、输出词表 logits
		src = self.pos(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim))
		tgt = self.pos(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim))
		# 生成目标端的因果掩码，防止看到未来词
		tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
		out = self.transformer(
			src,
			tgt,
			tgt_mask=tgt_mask,
			src_key_padding_mask=src_key_padding_mask,
			tgt_key_padding_mask=tgt_key_padding_mask,
			memory_key_padding_mask=src_key_padding_mask,
		)
		return self.fc(out)


def build_masks(src: torch.Tensor, tgt_in: torch.Tensor, pad_idx: int):
	# 生成源端和目标端的 padding mask
	src_pad = src.eq(pad_idx)
	tgt_pad = tgt_in.eq(pad_idx)
	return src_pad, tgt_pad


def train(model, loader, optimizer, criterion, device, pad_idx_src, pad_idx_tgt, epochs: int):
	# 训练循环：逐批更新参数
	model.train()
	for epoch in range(1, epochs + 1):
		total_loss = 0.0
		for src, tgt in loader:
			src = src.to(device)
			tgt = tgt.to(device)
			tgt_in = tgt[:, :-1]
			tgt_out = tgt[:, 1:]
			src_mask, tgt_mask = build_masks(src, tgt_in, pad_idx_src)
			logits = model(src, tgt_in, src_mask, tgt_mask)
			loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_loss += loss.item()
		print(f"epoch {epoch}/{epochs} loss={total_loss / max(len(loader), 1):.4f}")


@torch.no_grad()
def greedy_decode(model, src_tokens, src_vocab, tgt_vocab, inv_tgt_vocab, device, max_len: int = 40):
	# 贪心解码：每一步取概率最大的词
	model.eval()
	src = torch.tensor([encode(src_tokens, src_vocab, add_bos=True, add_eos=True)], dtype=torch.long, device=device)
	src_pad = src.eq(src_vocab[PAD])
	generated = [tgt_vocab[BOS]]
	for _ in range(max_len):
		tgt = torch.tensor([generated], dtype=torch.long, device=device)
		tgt_pad = tgt.eq(tgt_vocab[PAD])
		logits = model(src, tgt, src_pad, tgt_pad)
		next_id = int(logits[0, -1].argmax().item())
		if next_id == tgt_vocab[EOS]:
			break
		generated.append(next_id)
	tokens = [inv_tgt_vocab[idx] for idx in generated[1:] if idx in inv_tgt_vocab and inv_tgt_vocab[idx] not in {PAD, BOS, EOS}]
	return tokens


def evaluate(model, data, src_vocab, tgt_vocab, device):
	# 在开发集或测试集上计算 sacrebleu 的标准 BLEU4
	inv_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}
	references = []
	candidates = []
	for src_tokens, ref_tokens in data:
		pred_tokens = greedy_decode(model, src_tokens, src_vocab, tgt_vocab, inv_tgt_vocab, device)
		references.append(" ".join(ref_tokens))
		candidates.append(" ".join(pred_tokens))
	if not references or not candidates:
		return 0.0
	return float(sacrebleu.corpus_bleu(candidates, [references]).score)


def main():
	# 命令行入口：支持训练、评估和单句翻译
	parser = argparse.ArgumentParser(description="Simple Transformer MT")
	parser.add_argument("--mode", choices=["train", "eval", "translate"], default="train")
	parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
	parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--d-model", type=int, default=64)
	parser.add_argument("--nhead", type=int, default=4)
	parser.add_argument("--layers", type=int, default=1)
	parser.add_argument("--ffn-dim", type=int, default=128)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--max-len", type=int, default=40)
	parser.add_argument("--src", type=str, default="")
	args = parser.parse_args()

	# 检查数据文件是否存在，并且确实是压缩包
	data_path = Path(args.data)
	if not data_path.exists():
		raise FileNotFoundError(f"找不到数据文件: {data_path}")
	if data_path.suffixes[-2:] != [".tar", ".gz"]:
		raise ValueError("当前脚本默认读取 sample.tar.gz 数据包")

	# 读取数据并分别构建源语言和目标语言词表
	train_pairs, dev_pairs, test_src, test_ref = load_from_tar(data_path)
	src_vocab = build_vocab([src for src, _ in train_pairs])
	tgt_vocab = build_vocab([tgt for _, tgt in train_pairs])
	pad_idx_src = src_vocab[PAD]
	pad_idx_tgt = tgt_vocab[PAD]

	# 构建 DataLoader、模型、损失函数和优化器
	train_set = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
	loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate_fn(pad_idx_src, pad_idx_tgt))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = TransformerMT(len(src_vocab), len(tgt_vocab), args.d_model, args.nhead, args.layers, args.ffn_dim, args.dropout).to(device)
	weights_path = Path(args.weights)
	criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# 训练模式：训练后保存权重，并在开发集上做一次简单评估
	if args.mode == "train":
		train(model, loader, optimizer, criterion, device, pad_idx_src, pad_idx_tgt, args.epochs)
		torch.save({"model": model.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, weights_path)
		print(f"saved to {weights_path}")
		if dev_pairs:
			bleu = evaluate(model, dev_pairs, src_vocab, tgt_vocab, device)
			print(f"dev BLEU4: {bleu:.4f}")
		return

	# 非训练模式下先加载权重；如果没有权重就先训练一个 epoch
	if weights_path.exists():
		checkpoint = torch.load(weights_path, map_location=device)
		model.load_state_dict(checkpoint["model"])
		src_vocab = checkpoint.get("src_vocab", src_vocab)
		tgt_vocab = checkpoint.get("tgt_vocab", tgt_vocab)
	else:
		print("未找到权重，先训练 1 个 epoch")
		train(model, loader, optimizer, criterion, device, pad_idx_src, pad_idx_tgt, 1)

	# 评估模式：输出开发集和测试集 BLEU4
	if args.mode == "eval":
		if dev_pairs:
			bleu = evaluate(model, dev_pairs, src_vocab, tgt_vocab, device)
			print(f"dev BLEU4: {bleu:.4f}")
		if test_ref:
			preds = [greedy_decode(model, src, src_vocab, tgt_vocab, {idx: tok for tok, idx in tgt_vocab.items()}, device, args.max_len) for src in test_src]
			references = [" ".join(ref) for ref in test_ref]
			candidates = [" ".join(pred) for pred in preds]
			bleu = float(sacrebleu.corpus_bleu(candidates, [references]).score)
			print(f"test BLEU4: {bleu:.4f}")
		return

	# 翻译模式：输入一个分好词的源句，输出目标句
	if args.mode == "translate":
		if not args.src.strip():
			raise ValueError("translate 模式需要提供 --src")
		src_tokens = args.src.strip().split()
		inv_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}
		out = greedy_decode(model, src_tokens, src_vocab, tgt_vocab, inv_tgt_vocab, device, args.max_len)
		print(" ".join(out))


if __name__ == "__main__":
	# 直接运行脚本时进入主函数
	main()
