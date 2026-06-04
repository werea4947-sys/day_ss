import torch
!pip install torch torchtext transformers sacremoses sacrebleu -q
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
import sacrebleu
from tqdm import tqdm

#初始化环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#加载训练好的模型和tokenizer
model_name = "Helsinki-NLP/opus-mt-en-zh" 
tokenizer = MarianTokenizer.from_pretrained(model_name)

#初始化模型结构
model = MarianMTModel.from_pretrained(model_name)

#加载训练好的权重
model.load_state_dict(torch.load('/kaggle/input/best-1/best_model.pth', map_location=device))
model.to(device)
model.eval()  #切换到评估模式

#定义翻译函数
def translate(texts, batch_size=4, max_length=128):
    """
    批量翻译英文到中文
    :param texts: 英文句子列表，如 ["Hello world", "How are you?"]
    :return: 中文翻译列表
    """
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        #编码输入
        inputs = tokenizer(
            batch, 
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        #生成翻译
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=5,           
                early_stopping=True     #遇到<eos>停止
            )
        
        #解码输出
        batch_translations = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        translations.extend(batch_translations)
    
    return translations

#测试示例
if __name__ == "__main__":
    #示例句子
    test_sentences = [
        "Hello, how are you?",
        "Machine learning is fascinating."
    ]
    
    #执行翻译
    translated = translate(test_sentences)
    
    #打印结果
    for src, tgt in zip(test_sentences, translated):
        print(f"英文: {src}\n中文: {tgt}\n")