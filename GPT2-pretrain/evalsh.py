import transformers
import torch
import os
import json
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval1 import EvalDataset,evaluate_model


# step0: 读取参数
# Arguments
parser = argparse.ArgumentParser(description="Evaluate GPT-2 on custom dataset.")
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
parser.add_argument('--dataset_folder', type=str, default="./dataset/", help='Path to the dataset folder.')
parser.add_argument('--model_path', type=str, default="./models/", help='Path to the dataset folder.')
parser.add_argument('--result_path', type=str, default="./eval/", help='Path to the dataset folder.')
parser.add_argument('--file_single', action='store_true', help='Single file.')
parser.add_argument('--max_len', type=int,default=1024, help='Single file.')


args = parser.parse_args()
eval_path=args.dataset_folder
model_path=args.model_path
batch_size=args.batch_size
result_path=f'{args.result_path}/{model_path[-5:]}.json'
file_single=args.file_single
max_len=args.max_len


res=[]
eval_tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/xiaojin/hf_models/gpt2',padding_side='left')
eval_model = AutoModelForCausalLM.from_pretrained(model_path)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

many_gpu=False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count())
    eval_model = torch.nn.DataParallel(eval_model)
    many_gpu=True
eval_model.to(device)

if eval_path.endswith('json'):
    eval_dataset = EvalDataset(eval_tokenizer, eval_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    accuracy = evaluate_model(eval_model, eval_tokenizer, eval_dataloader, device,many_gpu)
    res.append(
                {
                    'model':model_path,
                    'eval_data':eval_path,
                    'acc':f'{accuracy:.4f}'
                }
            )
else:
    for filename in os.listdir(eval_path):
        # 检查文件是否以 .json 结尾
        if filename.endswith('.json'):
            file_path = os.path.join(eval_path, filename)
            eval_dataset = EvalDataset(eval_tokenizer, file_path)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            accuracy = evaluate_model(eval_model, eval_tokenizer, eval_dataloader, device,many_gpu)
            res.append(
                {
                    'model':model_path,
                    'eval_data':file_path,
                    'acc':f'{accuracy:.4f}'
                }
            )
        with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(res,json_file,ensure_ascii=False,indent=4)

with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(res,json_file,ensure_ascii=False,indent=4)





