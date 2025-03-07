import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
class GPT2AttentionRoPE():
    def __init__(self, config):
        super().__init__(config)
        self.inv_freq = torch.exp(torch.arange(0, self.head_dim, 2) * -(math.log(10000.0) / self.head_dim))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 新增RoPE位置编码
        seq_len = key.size(2)
        position_ids = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    

        angles = position_ids * self.inv_freq.unsqueeze(0)
        device = query.device
        cos = torch.cos(angles).to(device).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).to(device).unsqueeze(0).unsqueeze(0) 
        query_rot = torch.stack([query[..., ::2] * cos, query[..., 1::2] * sin], dim=-1).flatten(-2)
        key_rot = torch.stack([key[..., ::2] * cos, -key[..., 1::2]* sin], dim=-1).flatten(-2)

        #print(query_rot.shape)

        # 计算注意力权重
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key_rot = torch.cat((past_key, key_rot), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key_rot, value))


        attn_weights = torch.matmul(query_rot, key_rot.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output,self.num_heads,self.head_dim)
        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present) if use_cache else (attn_output,)
        if output_attentions:
            outputs += (attn_weights)

        return outputs


class GPT2ModelRoPE(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2AttentionRoPE(config) for _ in range(config.n_layer)])

class GPT2LMHeadModelRoPE(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelRoPE(config)
        self.init_weights()
        