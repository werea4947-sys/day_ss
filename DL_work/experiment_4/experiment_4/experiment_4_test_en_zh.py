import torch
!pip install torch torchtext transformers sacremoses sacrebleu -q
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
import sacrebleu
from tqdm import tqdm

# 1. 初始化环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 加载模型和分词器
def load_model(model_path, device):
    """加载预训练模型和训练好的权重"""
    # 注意：这里应该使用与训练时相同的模型名称
    model_name = "Helsinki-NLP/opus-mt-zh-en"  # 中文->英文模型
    
    # 加载官方tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # 初始化模型结构
    model = MarianMTModel.from_pretrained(model_name)
    
    # 加载训练好的权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 处理多GPU训练保存的权重（如果存在module前缀）
    if all(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 切换到评估模式
    
    return model, tokenizer

# 3. 改进的翻译函数
def translate(
    model, 
    tokenizer,
    texts, 
    batch_size=8, 
    max_length=128,
    beam_size=5,
    length_penalty=1.0,
    num_return_sequences=1
):
    translations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            
            # 编码输入
            inputs = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # 生成配置
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=beam_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=2  # 避免重复n-gram
            )
            
            # 解码输出
            batch_translations = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # 处理多候选情况
            if num_return_sequences > 1:
                batch_translations = [
                    batch_translations[i:i + num_return_sequences] 
                    for i in range(0, len(batch_translations), num_return_sequences)
                ]
            
            translations.extend(batch_translations)
    
    return translations

# 4. 测试案例
if __name__ == "__main__":
    # 加载模型（修改为你的实际路径）
    model, tokenizer = load_model("/kaggle/input/1111111/best_model_zh_en.pth", device)
    
    # 测试数据
    test_sentences = [
        "你好，最近怎么样？",
        "深度学习是人工智能的一个重要分支",
        "北京是中国的首都",
        "这家餐厅的招牌菜是什么？"
    ]
    
    # 执行翻译（常规模式）
    print("\n标准翻译:")
    translated = translate(model, tokenizer, test_sentences)
    for src, tgt in zip(test_sentences, translated):
        print(f"中文: {src}\n英文: {tgt}\n")
    
    # 多候选生成模式
    print("\nTop-3候选翻译:")
    candidates = translate(
        model, 
        tokenizer,
        test_sentences[:2],  # 只测试前两句
        num_return_sequences=3,
        beam_size=10
    )
    for i, src in enumerate(test_sentences[:2]):
        print(f"中文: {src}")
        for j, tgt in enumerate(candidates[i]):
            print(f"候选{j+1}: {tgt}")
        print()