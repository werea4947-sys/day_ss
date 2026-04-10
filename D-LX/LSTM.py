import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

# 加载训练数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# 数据预处理
class PoetryDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.tokenizer = self.build_tokenizer(text)
        self.data = self.prepare_sequences(text)

    def build_tokenizer(self, text):
        words = text.split()
        vocab = Counter(words)
        word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
        word_to_idx["<PAD>"] = 0
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return {"word_to_idx": word_to_idx, "idx_to_word": idx_to_word}

    def prepare_sequences(self, text):
        words = text.split()
        word_to_idx = self.tokenizer["word_to_idx"]
        sequences = []
        for i in range(len(words) - self.seq_length):
            seq = words[i:i + self.seq_length + 1]
            sequences.append([word_to_idx[word] for word in seq])
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[-1])

# 定义LSTM模型
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(PoetryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, hidden

# 生成诗句
def generate_poem(seed_text, model, tokenizer, seq_length, num_words=20, device="cpu"):
    model.eval()
    words = seed_text.split()
    word_to_idx = tokenizer["word_to_idx"]
    idx_to_word = tokenizer["idx_to_word"]

    for _ in range(num_words):
        input_seq = [word_to_idx.get(word, 0) for word in words[-seq_length:]]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model(input_seq)
            predicted_idx = torch.argmax(output, dim=1).item()
        words.append(idx_to_word[predicted_idx])
    return " ".join(words)

# 主程序
if __name__ == "__main__":
    # 替换为你的诗句数据文件路径
    filepath = "poems.txt"
    text = load_data(filepath)

    # 参数设置
    seq_length = 10
    embed_size = 128
    hidden_size = 256
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测 GPU

    # 数据集和数据加载器
    dataset = PoetryDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: pad_sequence([i[0] for i in x], batch_first=True, padding_value=0))

    vocab_size = len(dataset.tokenizer["word_to_idx"])
    model = PoetryModel(vocab_size, embed_size, hidden_size).to(device)  # 模型加载到 GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 模型训练
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # 数据加载到 GPU
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 测试生成诗句
    seed_text = "春风又绿江南岸"
    print("生成的诗句：")
    print(generate_poem(seed_text, model, dataset.tokenizer, seq_length, device=device))
