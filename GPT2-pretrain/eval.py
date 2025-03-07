import os
import re
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

# different with pretrain_gpt-2.py
class EvalDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=1024):
        self.tokenizer = tokenizer
        self.questions = []
        self.solutions = []
    
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data_list = json.load(json_file)
        #data_list= data_list[0:2]
        for data in tqdm(data_list,total=len(data_list),desc='processing'):
            question = f"{data['question']}"
            solution = data['solution']
            self.questions.append(question)
            self.solutions.append(solution)
        

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.solutions[idx]


def extract_last_number(text):
    numbers = re.findall(r'\b\d+\b', text)
    return int(numbers[-1]) if numbers else None


def evaluate_model(model, tokenizer, dataloader, device,many_gpu=True):
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        print("start!")
        for questions, solutions in tqdm(dataloader, desc="Evaluating"):
            inputs = tokenizer(questions, return_tensors='pt', padding=True, 
                               truncation=True, max_length=1024, padding_side='left').to(device)
            outputs=''
            if many_gpu:
                outputs = model.module.generate(**inputs, max_length=1024, num_return_sequences=1, do_sample=True,pad_token_id=tokenizer.eos_token_id, temperature=0.8, num_beams=5)
            else:
                outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1, do_sample=True,
                                        pad_token_id=tokenizer.eos_token_id, temperature=0.8, num_beams=5)

            generated_texts = tokenizer.batch_decode(outputs, q=True)
            
            print(len(generated_texts))
            print('*'*10)
            print(len(solutions))
            
            print(generated_texts[0])
            print('*'*10)
            print(solutions[0])
            for i in range(len(solutions)):
                generated_number = extract_last_number(generated_texts[i])
                solution_number = extract_last_number(solutions[i])

                if generated_number == solution_number:
                    correct_count += 1
                total_count += 1
            accuracy = correct_count /total_count
            print("current acc : ", accuracy)
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print('total acc',accuracy)
    return accuracy



    

