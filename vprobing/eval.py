#eval字段
# Additional imports
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    
# Evaluation function
def evaluate_model(model,tokenizer, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            text, labels = batch
            inputs = tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=1024, padding_side='left').to(device)
            labels = torch.tensor(labels).to(device)
        
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

#data
data=[]
task='nece'
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

task_data=task_data[:1000]
print('*'*100)
print('task',len(task_data))
dataset = ProbeDataset(task_data)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=4)

model_path='/mnt/workspace/xiaojin/physics/pretrain/models/medium'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path,output_hidden_states=True,return_dict_in_generate=True)
model.to(device)
config = AutoConfig.from_pretrained(model_path)
model = GPT2ForClassification(model,config,tokenizer,2).to(device)
#tokenizer.add_tokens(['[START]', '[END]'])
tokenizer.pad_token = tokenizer.eos_token
model.load_state_dict(torch.load(f'./models/epoch19_gpt2_medium_nece.pth'))
# 如果你想在推理时使用模型
model.eval()  # 切换到评估模式
accuracy = evaluate_model(model, tokenizer,dataloader, device)
print(f"Evaluation Accuracy: {accuracy:.4f}")
print('Evaluation finished!')
