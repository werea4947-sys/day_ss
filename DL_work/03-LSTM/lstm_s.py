import argparse
from pathlib import Path

import numpy as np  # 数值计算和加载 npz 数据
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # 数据集和数据加载器

try:  # 尝试导入 SwanLab
    import swanlab  # 在线/离线实验记录工具
except ImportError:  # 如果没有安装就降级处理
    swanlab = None  


DATA_PATH = Path(__file__).with_name("实验3：数据集tang.npz")  # 数据集路径
DEFAULT_WEIGHTS = Path(__file__).with_name("poetry_lstm.pth")  # 默认权重文件路径

class Config:  # 统一放训练参数
    batch_size = 64  # 批大小
    embed_dim = 128  # 词向量维度
    hidden_dim = 512  # LSTM 隐藏层维度
    num_layers = 2  # LSTM 层数
    dropout = 0.2  # dropout 比例
    lr = 3e-4  # 学习率
    eta_min = 1e-6  # 余弦退火最小学习率
    scheduler = "none"  # 学习率调度器: none/cosine
    log_interval = 50  # 每隔多少个 batch 记录一次 step 日志
    epochs = 50  # 训练轮数
    max_gen_len = 125  # 最长生成长度
    top_k = 5  # 采样时保留前k个候选

class PoetryDataset(Dataset):  # 诗词数据集封装
    def __init__(self, data):  # 初始化
        self.data = torch.from_numpy(data.astype(np.int64))  # 转成 长Tensor

    def __len__(self):  # 数据集长度
        return len(self.data)  # 返回样本数

    def __getitem__(self, index):  # 取一个样本
        sample = self.data[index]  # 取第 index 首诗
        return sample[:-1], sample[1:]  # 输入为前半段，标签为后半段

class PoetryModel(nn.Module):  # 诗歌生成模型
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):  # 定义网络结构
        super().__init__()  # 调用父类初始化
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层
        self.embedding_dropout = nn.Dropout(dropout)  # 嵌入后加 dropout
        self.lstm = nn.LSTM(  # 堆叠式 LSTM
            embed_dim,  # 输入维度
            hidden_dim,  # 隐藏维度
            num_layers=num_layers,  # LSTM 层数
            dropout=dropout if num_layers > 1 else 0.0,  # 多层时才在层间 dropout
            batch_first=True,  # 输入形状为   [batch, seq, feature]
        )  # LSTM 结束
        self.output_norm = nn.LayerNorm(hidden_dim)  #输出层归一化
        self.dropout = nn.Dropout(dropout)  #输出再做一次 dropout
        self.fc = nn.Linear(hidden_dim, vocab_size)  #映射到词表大小

    def forward(self, x, hidden=None):  # 前向传播
        x = self.embedding(x)  # 先查词向量
        x = self.embedding_dropout(x)  # 再随机失活一部分特征
        out, hidden = self.lstm(x, hidden)  # 经过 LSTM
        out = self.output_norm(out)  # 做归一化稳定训练
        out = self.dropout(out)  # 再做一次 dropout
        out = self.fc(out)  # 输出每个位置的词表 logits
        return out, hidden  # 返回输出和隐状态

def load_data(path=DATA_PATH):  # 读取数据集
    data = np.load(path, allow_pickle=True)  # 加载 npz 文件
    return data["data"], data["ix2word"].item(), data["word2ix"].item()  # 返回三部分内容

def build_model(word2ix):  # 根据词表构建模型
    return PoetryModel(  # 返回模型实例
        len(word2ix),  # 词表大小
        Config.embed_dim,  # 嵌入维度
        Config.hidden_dim,  # 隐藏层维度
        Config.num_layers,  # LSTM 层数
        Config.dropout,  # dropout 比例
    )  # 模型构建结束


def train_model(model, loader, word2ix, device, use_swanlab=False):  # 训练模型
    pad_idx = word2ix["</s>"]  # 获取填充符的下标
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # 忽略填充符位置的损失
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)  # 使用 Adam 优化器
    scheduler = None  # 默认不使用调度器
    if Config.scheduler == "cosine":  # 可选余弦退火
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.epochs,
            eta_min=Config.eta_min,
        )

    model.train()  # 切换到训练模式
    global_step = 0  # 全局训练步数
    for epoch in range(Config.epochs):  # 遍历每个 epoch
        total_loss = 0.0  # 统计 epoch 总损失
        epoch_correct = 0  # 统计 epoch 正确 token 数
        epoch_tokens = 0  # 统计 epoch 有效 token 数
        for x, y in loader:  # 遍历每个 batch
            x = x.to(device)  # 输入放到设备上
            y = y.to(device)  # 标签放到设备上
            logits, _ = model(x)  # 前向传播得到预测
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # 计算交叉熵损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止爆炸
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加 batch 损失

            with torch.no_grad():  # 统计 token 准确率，不参与梯度
                pred = logits.argmax(dim=-1)  # 取每个位置预测概率最大的字
                valid_mask = y != pad_idx  # 忽略填充位
                correct = ((pred == y) & valid_mask).sum().item()  # 正确预测的有效 token 数
                tokens = valid_mask.sum().item()  # 有效 token 总数
                epoch_correct += correct  # 累加到 epoch 统计
                epoch_tokens += tokens  # 累加到 epoch 统计

            global_step += 1  # 更新全局步数
            if use_swanlab and swanlab is not None and global_step % Config.log_interval == 0:  # step 级日志
                step_acc = (correct / max(tokens, 1)) if tokens > 0 else 0.0  # 当前 batch token 准确率
                swanlab.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/token_acc_step": step_acc,
                        "train/lr_step": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "epoch": epoch + 1,
                    }
                )
        avg_loss = total_loss / len(loader)  # 计算平均损失
        epoch_acc = (epoch_correct / max(epoch_tokens, 1)) if epoch_tokens > 0 else 0.0  # epoch token 准确率
        perplexity = float(np.exp(min(avg_loss, 20.0)))  # 由交叉熵近似计算困惑度
        current_lr = optimizer.param_groups[0]["lr"]  # 当前学习率
        print(
            f"epoch {epoch + 1}/{Config.epochs}, loss={avg_loss:.4f}, "
            f"token_acc={epoch_acc:.4f}, ppl={perplexity:.2f}, lr={current_lr:.6g}"
        )  # 打印训练进度
        if use_swanlab and swanlab is not None:  # 如果启用了 SwanLab
            swanlab.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/token_acc_epoch": epoch_acc,
                    "train/ppl_epoch": perplexity,
                    "train/lr_epoch": current_lr,
                    "epoch": epoch + 1,
                }
            )  # 记录 epoch 级指标

        if scheduler is not None:  # 每个 epoch 后更新学习率
            scheduler.step()


def sample_topk(logits, top_k=5, temperature=1.0):  # 从 top-k 中采样一个词
    logits = logits / max(temperature, 1e-6)  # 温度缩放，避免除零
    top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)  # 取前 k 个候选
    probs = torch.softmax(top_vals, dim=-1)  # 转成概率
    choice = torch.multinomial(probs, 1)  # 按概率随机采样
    return top_idx.gather(-1, choice).item()  # 返回采样到的词下标


def generate(model, start_words, ix2word, word2ix, device):  # 根据首句生成诗句
    model.eval()  # 切换到推理模式
    start_words = start_words.strip()  # 去掉首尾空格
    result = list(start_words)  # 把首句拆成字符列表
    start_idx = word2ix["<START>"]  # 获取起始符下标
    hidden = None  # 初始隐状态为空
    input_ids = torch.tensor([[start_idx]], dtype=torch.long, device=device)  # 先输入起始符
    with torch.no_grad():  # 关闭梯度计算
        output, hidden = model(input_ids, hidden)  # 先跑一遍起始符
        for char in start_words:  # 依次喂入首句字符
            if char not in word2ix:  # 如果字符不在词表中
                continue  # 跳过未知字符
            input_ids = torch.tensor([[word2ix[char]]], dtype=torch.long, device=device)  # 转成下标输入
            output, hidden = model(input_ids, hidden)  # 更新隐状态
        for _ in range(Config.max_gen_len):  # 继续生成若干步
            next_idx = sample_topk(output[:, -1, :], top_k=Config.top_k)  # 采样下一个字
            next_word = ix2word[next_idx]  # 将下标转成汉字
            if next_word == "<EOP>":  # 如果遇到结束符
                break  # 结束生成
            if next_word != "</s>":  # 如果不是填充符
                result.append(next_word)  # 加入结果
            input_ids = torch.tensor([[next_idx]], dtype=torch.long, device=device)  # 继续把预测字喂回模型
            output, hidden = model(input_ids, hidden)  # 更新输出和隐状态
    return "".join(result)  # 把字符列表拼成字符串


def main():  # 主函数入口
    parser = argparse.ArgumentParser(description="Simple poetry LSTM")  # 创建参数解析器
    parser.add_argument("--mode", choices=["train", "generate"], default="generate")  # 选择训练或生成
    parser.add_argument("--start", type=str, default="湖光秋月两相和")  # 生成时的首句
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))  # 权重文件路径
    parser.add_argument("--epochs", type=int, default=Config.epochs)  # 可覆盖训练轮数
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)  # 可覆盖 batch 大小
    parser.add_argument("--lr", type=float, default=Config.lr)  # 可覆盖学习率
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)  # 可覆盖隐藏层维度
    parser.add_argument("--num-layers", type=int, default=Config.num_layers)  # 可覆盖 LSTM 层数
    parser.add_argument("--dropout", type=float, default=Config.dropout)  # 可覆盖 dropout
    parser.add_argument("--scheduler", type=str, default=Config.scheduler, choices=["none", "cosine"])  # 学习率调度器
    parser.add_argument("--eta-min", type=float, default=Config.eta_min)  # 余弦退火最小学习率
    parser.add_argument("--log-interval", type=int, default=Config.log_interval)  # step 日志间隔
    parser.add_argument("--use-swanlab", action="store_true")  # 是否启用 SwanLab
    parser.add_argument("--swanlab-project", type=str, default="lstm-poetry")  # SwanLab 项目名
    parser.add_argument("--swanlab-run-name", type=str, default="")  # SwanLab 实验名
    parser.add_argument("--swanlab-mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])  # SwanLab 模式
    args = parser.parse_args()  # 解析参数
    Config.epochs = args.epochs  # 让命令行轮数覆盖默认值
    Config.batch_size = args.batch_size  # 覆盖 batch 大小
    Config.lr = args.lr  # 覆盖学习率
    Config.hidden_dim = args.hidden_dim  # 覆盖隐藏层维度
    Config.num_layers = args.num_layers  # 覆盖 LSTM 层数
    Config.dropout = args.dropout  # 覆盖 dropout
    Config.scheduler = args.scheduler  # 覆盖调度器
    Config.eta_min = args.eta_min  # 覆盖最小学习率
    Config.log_interval = max(1, args.log_interval)  # 限制日志间隔至少为 1
    data, ix2word, word2ix = load_data()  # 加载诗词数据和映射
    dataset = PoetryDataset(data)  # 构建数据集对象
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)  # 构建数据加载器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 CPU 或 GPU
    model = build_model(word2ix).to(device)  # 构建模型并放到设备上
    weights_path = Path(args.weights)  # 转成 Path 对象
    use_swanlab = args.use_swanlab and args.mode == "train" and args.swanlab_mode != "disabled"  # 判断是否启用 SwanLab
    if use_swanlab:  # 如果需要记录实验
        if swanlab is None:  # 如果 SwanLab 没安装
            raise ImportError("SwanLab 未安装，请先执行: pip install swanlab")  # 抛出安装提示
        init_kwargs = {  # 组织 SwanLab 初始化参数
            "project": args.swanlab_project,  # 项目名
            "mode": args.swanlab_mode,  # 云端、离线等模式
            "config": {  # 实验配置
                "batch_size": Config.batch_size,  # 批大小
                "embed_dim": Config.embed_dim,  # 嵌入维度
                "hidden_dim": Config.hidden_dim,  # 隐藏层维度
                "num_layers": Config.num_layers,  # 层数
                "dropout": Config.dropout,  # dropout
                "lr": Config.lr,  # 学习率
                "scheduler": Config.scheduler,  # 学习率调度器
                "eta_min": Config.eta_min,  # 余弦最小学习率
                "log_interval": Config.log_interval,  # step 日志间隔
                "epochs": Config.epochs,  # 训练轮数
            },  # 配置结束
        }  # 初始化参数结束
        if args.swanlab_run_name:  # 如果指定了实验名
            init_kwargs["experiment_name"] = args.swanlab_run_name  # 写入实验名
        swanlab.init(**init_kwargs)  # 启动 SwanLab run
    if args.mode == "train":  # 如果是训练模式
        train_model(model, loader, word2ix, device, use_swanlab=use_swanlab)  # 训练模型
        torch.save(model.state_dict(), weights_path)  # 保存模型权重
        print(f"saved to {weights_path}")  # 打印保存位置
    else:  # 如果是生成模式
        if weights_path.exists():  # 如果权重文件存在
            model.load_state_dict(torch.load(weights_path, map_location=device))  # 加载已有权重
        else:  # 如果没有权重文件
            train_model(model, loader, word2ix, device)  # 先简单训练
            torch.save(model.state_dict(), weights_path)  # 再保存权重
        poem = generate(model, args.start, ix2word, word2ix, device)  # 生成诗句
        print(poem)  # 输出生成结果
    if use_swanlab and swanlab is not None and hasattr(swanlab, "finish"):  # 如果需要收尾
        swanlab.finish()  # 结束 SwanLab run

if __name__ == "__main__":  # 脚本入口判断
    main()  # 调用主函数
