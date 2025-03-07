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

model_path='/mnt/workspace/xiaojin/physics/pretrain/models/gpt2_hard_epoch1'
tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/xiaojin/hf_models/gpt2', padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path,output_hidden_states=True,return_dict_in_generate=True)
config = AutoConfig.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

text='hellod'
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024, padding_side='left')
input_ids=inputs['input_ids'].to(device)
attention_mask=inputs['attention_mask'].to(device)


model.to(device)
#outputs = model.generate(**input_ids, max_length=1024, num_return_sequences=1, do_sample=True,
                                        #pad_token_id=tokenizer.eos_token_id, temperature=0.8, num_beams=1)

outputs=model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#inputs=inputs.to(device)
#outputs = self.model.generate(**inputs, max_length=1024, num_return_sequences=1, do_sample=True,
                                        #pad_token_id=self.tokenizer.eos_token_id, temperature=0.8, num_beams=1)

hidden_states = outputs.hidden_states


print(hidden_states[-1].mean(dim=1))


            