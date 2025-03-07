import numpy as np
from datetime import datetime
from torch.nn import DataParallel
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
import argparse
from tqdm import tqdm
import os
import json

# Arguments
parser = argparse.ArgumentParser(description="Finetune GPT-2 for V-probing task.")
parser.add_argument('--model_path', type=str, default='/mnt/workspace/xiaojin/hf_models/gpt2', help='Number of training epochs.')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
parser.add_argument('--rank', type=int, default=8, help='low-rank update.')
parser.add_argument('--gradient_accumulation', type=int, default=8, help='low-rank update.')
parser.add_argument('--label_num', type=int, default=2, help='classification label num.')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer.')
parser.add_argument('--save_model_name', type=str, default="./models/", help='saved model name.')
parser.add_argument('--task', type=str, default="nece", help='saved model name.')
args = parser.parse_args()

rank = args.rank
task=args.task
model_path=args.model_path
num_labels = args.label_num
gradient_accumulation=args.gradient_accumulation

# step1: build a dataset class
# 处理特殊的输入tokens
def process_input(text, tokenizer,special_tokens=['[START]', '[END]']):
    tokenizer.add_tokens(special_tokens)
    inputs = tokenizer(text, return_tensors='pt', padding=True,
                       truncation=True, max_length=1024, padding_side='left')
    return inputs

#整理数据集
class ProbeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        return text, label

#step2: low-rank update
def low_rank_update(embedding_layer, rank):
    original_weights = embedding_layer.weight.data
    # 检查并修复非有限值
    if not torch.isfinite(original_weights).all():
        original_weights = torch.nan_to_num(original_weights)

    u, s, v = torch.svd(original_weights)
    u_low_rank = u[:, :rank]
    s_low_rank = s[:rank]
    v_low_rank = v[:, :rank]
    low_rank_weights = u_low_rank @ torch.diag(s_low_rank) @ v_low_rank.t()
    embedding_layer.weight = nn.Parameter(low_rank_weights)

# step3: classification for adding a linear layer 
class GPT2ForClassification(nn.Module):
    def __init__(self, model,config,tokenizer, num_labels):
        super(GPT2ForClassification, self).__init__()
        self.model = model
        self.tokenizer=tokenizer
        self.classifier = nn.Linear(config.hidden_size, num_labels).to(device)

    def forward(self, inputs, attention_mask=None):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.hidden_states

        #最后一层token的均值
        pooled_output = hidden_states[-1].mean(dim=1).to(device)
        logits = self.classifier(pooled_output)
        return logits

# step1: load model
# 加载预训练的GPT-2模型和分词器
model_path='/mnt/workspace/xiaojin/physics/pretrain/models/medium'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path,output_hidden_states=True,return_dict_in_generate=True)
config = AutoConfig.from_pretrained(model_path)
#tokenizer.add_tokens(['[START]', '[END]'])
tokenizer.pad_token = tokenizer.eos_token

# 初始化带 ROPE 的 GPT2 模型
#model = GPT2LMHeadModel.from_pretrained(model_path, output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
low_rank_update(model.transformer.wte, rank)
classification_model = GPT2ForClassification(model,config,tokenizer,num_labels).to(device)

data=[]
directory='./data/'
for filename in os.listdir(directory):
    # 检查文件名是否以 'nece' 开头且以 '.json' 结尾
    if filename.startswith(task) and filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data_cur = json.load(file)
                data.extend(data_cur)
                print(f"内容来自 {filename}\n")
            except json.JSONDecodeError as e:
                print(f"无法解析 {filename}: {e}")

#data=data[:1000]
task_data=[]
if task=='nece':
    for d in data:
        for j in d['nece']:
            try:
                task_data.append(
                {"text":d['question']+'[START]'+j['variable']+'[END]',
                "label":j['judgement']}
                )
            except:
                continue

task_data=task_data[:10000]
print(len(task_data))

dataset = ProbeDataset(task_data)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classification_model.classifier.parameters(), lr=args.learning_rate)

# 训练循环
classification_model.train()
wandb.init(project='gpt2_vprobing', name='probing')
wandb.watch(classification_model, log="all")

model_name = args.save_model_name
for epoch in range(args.epochs): 
    for step,batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        text, labels = batch
        inputs = tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=1024, padding_side='left')
        
        print('config',config.hidden_size)
        try:
            logits = classification_model(inputs)
        except:
            continue
        loss = criterion(logits, torch.tensor(labels).to(device))
        if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
        loss.backward()
        #  optimizer step
        if (step + 1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
        wandb.log({"loss":loss.item(),
               "epoch":epoch})
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    if (epoch+1)%1==0:
        print('training finished!')
        # 保存模型的状态字典
        torch.save(classification_model.state_dict(), f'./models/epoch{epoch}_{model_name}')


    




