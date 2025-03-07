import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, get_linear_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
from tqdm import tqdm
import random
import numpy as np
import wandb
from torch.distributed.elastic.multiprocessing.errors import record


@record
# dataset
class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=1024):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data_list = json.load(json_file)
        for data in tqdm(data_list):
            #data=data['content']
            question = data['question']
            solution = data['solution']
            combined_text = question + ' ' + solution
            encodings_dict = tokenizer(combined_text,
            truncation=True,max_length=max_length,padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


# Arguments
parser = argparse.ArgumentParser(description="Pre-train GPT-2 on custom dataset.")
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer.')
parser.add_argument('--dataset_folder', type=str, default="./dataset/", help='Path to the dataset folder.')
args = parser.parse_args()

# Initialize tokenizer and model from scratch
tokenizer = AutoTokenizer.from_pretrained('/cpfs01/user/xiaojin/xiaojin/hf_models/gpt2')  # Initialize from gpt2 vocab
# pre-training
# 模型配置
config = GPT2Config()  # Default config (new weights)
config.n_ctx=1024
config.n_positions=1024
#model = AutoModelForCausalLM.from_config(config)  # Initialize model from scratch
model = AutoModelForCausalLM.from_pretrained('/cpfs01/user/xiaojin/xiaojin/physics/pretrain/models/xiaojin/model_output_25')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset and DataLoader
# 数据集
dataset = MyDataset(tokenizer, args.dataset_folder)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
total_steps = len(dataloader) * args.epochs
print('total steps',total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Training loop

model.train()
wandb.init(project='gpt2_xiaojin', name='pretrain')
wandb.watch(model, log="all")
print("****** start ******", flush=True)
for epoch in range(0, args.epochs):
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
        losses = []
        for step, (batch_input_ids, batch_attn_masks) in enumerate(dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_attn_masks = batch_attn_masks.to(device)
            
            outputs = model(batch_input_ids, attention_mask=batch_attn_masks, labels=batch_input_ids)
            loss = outputs.loss
            loss=loss.mean()
            loss.backward()
            wandb.log({"loss":loss,
               "step":step,
               "epoch":epoch})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            # tqdm.write(f"Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            losses.append(loss.item())
        if (epoch+1)%1==0:
            output_dir = f"../models/xiaojin/model_output_new_{epoch+1}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.module.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

