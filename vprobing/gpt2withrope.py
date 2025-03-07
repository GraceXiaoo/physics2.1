import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from rotary_embedding_torch import RotaryEmbedding
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RotaryGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.rotary_emb = RotaryEmbedding(dim=config.n_embd // config.n_head, use_xpos=True)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Apply rotary embeddings with XPos
        query, key = self.rotary_emb.rotate_queries_and_keys(query, key)
        
        # Continue with the original attention mechanism
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

class ModifiedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.wpe = torch.nn.Embedding(config.n_positions, config.hidden_size)
        self.h = torch.nn.ModuleList([RotaryGPT2Attention(config) for _ in range(config.n_layer)])
        self.init_weights()
