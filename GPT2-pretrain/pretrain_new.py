import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
import os
import json
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# RoPE实现
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):  # 限制 max_position_embeddings
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self.cos_cached = None
        self.sin_cached = None
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=1024):  # 限制 max_length
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        with open(file_path, 'r', encoding='utf-8') as json_file:
            data_list = json.load(json_file)
            for data in tqdm(data_list, desc="Loading dataset"):
                question = data['question']
                solution = data['solution']
                combined_text = question + ' ' + solution

                encodings_dict = tokenizer(
                    combined_text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )

                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# 主训练函数
def main():
    parser = argparse.ArgumentParser(description="Pre-train GPT-2 with RoPE on custom dataset.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--accumulation_steps', type=int, default=2)  # 添加梯度累积
    args = parser.parse_args()

    # 初始化分布式训练环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 初始化tokenizer
    model_path='/cpfs01/user/xiaojin/xiaojin/physics/pretrain/models/xiaojin/model_output_25'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化模型配置和模型
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    #new_max_length=2048
    #config = GPT2Config.from_pretrained(model_path, n_positions=new_max_length, n_ctx=new_max_length)
    #model = GPT2LMHeadModel(config)

    # 替换注意力层
    for layer in model.transformer.h:
        layer.attn.rotary_emb = RotaryEmbedding(layer.attn.head_dim, max_position_embeddings=config.n_positions)

    # 数据加载
    dataset = MyDataset(tokenizer, args.dataset_path, max_length=args.max_seq_length)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # 模型分布式包装
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 添加预热
        num_training_steps=total_steps
    )

    # 训练循环
    model.train()
    print(f"****** Training Started on GPU {local_rank} ******", flush=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0  # 记录每个 epoch 的损失
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1} (GPU {local_rank})") as pbar:
            for step, (batch_input_ids, batch_attn_masks) in enumerate(dataloader):
                batch_input_ids = batch_input_ids.to(device)
                batch_attn_masks = batch_attn_masks.to(device)

                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attn_masks,
                    labels=batch_input_ids
                )

                loss = outputs.loss / args.accumulation_steps  # 梯度累积
                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=loss.item() * args.accumulation_steps)
                pbar.update(1)

        if local_rank == 0 and (epoch+1)%2==0:
            epoch_output_dir = os.path.join(args.model_save_path, f"epoch_{epoch+1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            model.module.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)

        # 清理显存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
