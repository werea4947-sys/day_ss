import torch
import torch.nn as nn
!pip install torch torchtext transformers sacremoses sacrebleu -q
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW 
from transformers import get_linear_schedule_with_warmup
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from tqdm import tqdm
import os

#检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.device_count() >= 2:
    print(f"Using {torch.cuda.device_count()} GPUs!")

#参数配置
MAX_LEN = 128
BATCH_SIZE = 64 
EPOCHS = 30
LR = 5e-5
GRAD_ACCUM_STEPS = 2  

#数据加载
class TranslationDataset(Dataset):
    def __init__(self, zh_path, en_path, zh_tokenizer, en_tokenizer):
        with open(zh_path, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f]
        with open(en_path, 'r', encoding='utf-8') as f:
            self.en_lines = [line.strip() for line in f]
        
        self.zh_tokenizer = zh_tokenizer
        self.en_tokenizer = en_tokenizer

    def __len__(self):
        return len(self.zh_lines)

    def __getitem__(self, idx):
        zh_text = self.zh_lines[idx]
        en_text = self.en_lines[idx]
        
        zh_enc = self.zh_tokenizer(zh_text, 
                                 max_length=MAX_LEN,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        
        en_enc = self.en_tokenizer(en_text,
                                 max_length=MAX_LEN,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        
        return {
            'input_ids': zh_enc['input_ids'].squeeze(),
            'attention_mask': zh_enc['attention_mask'].squeeze(),
            'labels': en_enc['input_ids'].squeeze()
        }

#初始化Tokenizer和Model
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

#多GPU支持
if torch.cuda.device_count() >= 2:
    model = nn.DataParallel(model)
model.to(device)

#数据加载
dataset = TranslationDataset('/kaggle/input/zh-and-en/chinese.txt', '/kaggle/input/zh-and-en/english.txt', tokenizer, tokenizer)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

#优化器和调度器
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps=0.1*total_steps,
                                           num_training_steps=total_steps)

#训练和验证循环
def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        
        loss = outputs.loss.mean()  #多GPU取平均
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def evaluate(model, loader, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Eval]", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          labels=batch['labels'])
            
            loss = outputs.loss.mean()
            total_loss += loss.item()
            
            #生成预测
            preds = model.module.generate(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        max_length=MAX_LEN)
            
            #解码文本用于BLEU计算
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            
            all_preds.extend(pred_texts)
            all_labels.extend([[ref] for ref in label_texts])  #sacrebleu需要[[ref1], [ref2]]
    
    #计算BLEU-4（改为英文tokenization方式）
    bleu = sacrebleu.corpus_bleu(all_preds, all_labels, tokenize='13a')
    
    return total_loss / len(loader), bleu.score

#主训练循环
best_bleu = 0
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
    val_loss, bleu_score = evaluate(model, val_loader, epoch)
    
    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  BLEU-4: {bleu_score:.2f}")
    
    #保存最佳模型
    if bleu_score > best_bleu:
        best_bleu = bleu_score
        torch.save(model.module.state_dict(), 'best_model_zh_en.pth')
        print(f"  New best BLEU! Saved model.")
    
    #提前停止条件
    if bleu_score > 14:
        print(f"BLEU-4 reached {bleu_score:.2f} (>14), stopping training.")
        break

print(f"Training complete. Best BLEU-4: {best_bleu:.2f}")