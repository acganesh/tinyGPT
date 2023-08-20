import math
import sys

from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_encoder_hparams_and_params

def get_state_dict(params):
    data = {
        'layer_norm.b': torch.tensor(params['ln_f']['b']),
        'layer_norm.g': torch.tensor(params['ln_f']['g']),
        'wpe.weight': torch.tensor(params['wpe']),
        'wte.weight': torch.tensor(params['wte']),
    }

    for i, block in enumerate(params['blocks']):
        data[f"blocks.{i}.mha.linear1.weight"] = torch.tensor(block['attn']['c_attn']['w']).T
        data[f"blocks.{i}.mha.linear1.bias"] = torch.tensor(block['attn']['c_attn']['b'])
        data[f"blocks.{i}.mha.linear2.weight"] = torch.tensor(block['attn']['c_proj']['w']).T
        data[f"blocks.{i}.mha.linear2.bias"] = torch.tensor(block['attn']['c_proj']['b'])
        data[f"blocks.{i}.ffn.linear1.weight"] = torch.tensor(block['mlp']['c_fc']['w']).T
        data[f"blocks.{i}.ffn.linear1.bias"] = torch.tensor(block['mlp']['c_fc']['b'])
        data[f"blocks.{i}.ffn.linear2.weight"] = torch.tensor(block['mlp']['c_proj']['w']).T
        data[f"blocks.{i}.ffn.linear2.bias"] = torch.tensor(block['mlp']['c_proj']['b'])
        data[f"blocks.{i}.layer_norm1.g"] = torch.tensor(block['ln_1']['g'])
        data[f"blocks.{i}.layer_norm1.b"] = torch.tensor(block['ln_1']['b'])
        data[f"blocks.{i}.layer_norm2.g"] = torch.tensor(block['ln_2']['g'])
        data[f"blocks.{i}.layer_norm2.b"] = torch.tensor(block['ln_2']['b'])

    return data

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

class LayerNorm(nn.Module):
    def __init__(self, g, b, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(data=torch.tensor(g))
        self.b = nn.Parameter(data=torch.tensor(b))
        self.eps = eps
    
    def forward(self, x):
        mean = torch.mean(x, axis=-1, keepdim=True)
        variance = torch.var(x, axis=-1, keepdim=True)
        return self.g * (x - mean) / torch.sqrt(variance + self.eps) + self.b

class FeedForwardNetwork(nn.Module):
    def __init__(self, c_fc, c_proj):
        super().__init__()

        hidden_size, inner_size = c_fc['w'].shape
        inner_size, hidden_size = c_proj['w'].shape

        self.linear1 = nn.Linear(hidden_size, inner_size)
        self.linear2 = nn.Linear(inner_size, hidden_size)

        linear1_state_dict = {
            "weight": c_fc['w'],
            "bias": c_fc['b']
        }

        linear2_state_dict = {
            "weight": c_proj['w'],
            "bias": c_proj['b']
        }

        self.gelu = GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, c_attn, c_proj, n_head):
        super().__init__()

        linear1_in, linear1_out = c_attn['w'].shape
        linear2_in, linear2_out = c_proj['w'].shape

        self.n_head = n_head

        self.linear1 = nn.Linear(*c_attn['w'].shape)
        self.linear2 = nn.Linear(*c_proj['w'].shape)

    def _causal_mask(self, size):
        return (1 - torch.tril(torch.ones(size, size))) * -1e10

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.linear1(x)
        q, k, v = torch.chunk(x, 3, dim=-1)

        def reshape_qkv(tensor):
            return tensor.view(batch_size, seq_len, self.n_head, -1).permute(0, 2, 1, 3).contiguous()
        
        q, k, v = map(reshape_qkv, (q, k, v))

        causal_mask = self._causal_mask(seq_len)

        dk = k.size(-1)
        scores = q @ k.permute(0, 1, 3, 2) / math.sqrt(dk) + causal_mask
        x = F.softmax(scores, dim=-1) @ v

        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        x = self.linear2(x)
        return x

        




class TransformerBlock(nn.Module):
    def __init__(self, mlp, attn, ln_1, ln_2, n_head):
        super().__init__()
        self.mha = MultiHeadAttention(**attn, n_head=n_head)
        self.ffn = FeedForwardNetwork(**mlp)
        self.layer_norm1 = LayerNorm(**ln_1)
        self.layer_norm2 = LayerNorm(**ln_2)
    
    def forward(self, x):
        x = x + self.mha(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, wte, wpe, blocks, ln_f, n_head):
        super().__init__()

        vocab_size, embed_size = wte.shape
        context_length, embed_size = wpe.shape

        self.wte = nn.Embedding(vocab_size, embed_size)
        self.wpe = nn.Embedding(context_length, embed_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(block['mlp'],
                             block['attn'],
                             block['ln_1'],
                             block['ln_2'],
                             n_head=n_head) for block in blocks])

        self.layer_norm = LayerNorm(ln_f['g'], ln_f['b'])

    def forward(self, inputs):
        inputs = torch.tensor(inputs)
        inputs = inputs.unsqueeze(0)
        x = self.wte(inputs) + self.wpe(torch.arange(inputs.shape[1]))
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        res = x @ torch.unsqueeze(self.wte.weight.T, 0)
        return res

def generate(inputs, params, n_head, n_tokens_to_generate):
    gpt2 = GPT2(**params, n_head=n_head)
    state_dict = get_state_dict(params)
    gpt2.load_state_dict(state_dict)

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs)
        next_id = torch.argmax(logits[0, -1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    prompt = sys.argv[1]
    output_text = main(prompt=prompt)
    print(output_text)
