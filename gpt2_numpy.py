"""
Adapted from Jay Mody's GPT2: https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py
with some changes to improve readability.
"""

from utils import load_encoder_hparams_and_params
import math
import numpy as np

import sys

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    x = linear(x,
               w=c_fc['w'],
               b=c_fc['b'])
    x = gelu(x)
    x = linear(x,
               w=c_proj['w'],
               b=c_proj['b'])
    return x

def attention(q, k, v, causal_mask):
    d_k = k.shape[-1]
    return softmax(q @ k.T / math.sqrt(d_k) + causal_mask) @ v

def multi_head_attention(x, c_attn, c_proj, n_head):
    x = linear(x,
               w=c_attn['w'],
               b=c_attn['b'])

    qkv = np.split(x, 3, axis=-1)

    qkv_heads = []
    for elt in qkv:
        qkv_head_split = np.split(elt, n_head, axis=-1)
        qkv_heads.append(qkv_head_split)

    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    out_heads = []
    for q, k, v in zip(*qkv_heads):
        x = attention(q, k, v, causal_mask)
        out_heads.append(x)
    
    x = np.hstack(out_heads)

    x = linear(x,
               w=c_proj['w'],
               b=c_proj['b'])

    return x



def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + multi_head_attention(layer_norm(x, g=ln_1['g'], b=ln_1['b']),
                                 c_attn=attn['c_attn'],
                                 c_proj=attn['c_proj'],
                                 n_head=n_head)
    x = x + ffn(layer_norm(x, g=ln_2['g'], b=ln_2['b']),
                c_fc=mlp['c_fc'],
                c_proj=mlp['c_proj'])
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    import pdb; pdb.set_trace()

    for block in blocks:
        x = transformer_block(x,
                              mlp=block['mlp'],
                              attn=block['attn'],
                              ln_1=block['ln_1'],
                              ln_2=block['ln_2'],
                              n_head=n_head)
    
    x = layer_norm(x,
                   g=ln_f['g'],
                   b=ln_f['b'])
    res = x @ wte.T
    return res


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs,
                      wte=params['wte'],
                      wpe=params['wpe'],
                      blocks=params['blocks'],
                      ln_f=params['ln_f'],
                      n_head = n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate :]


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "1558M", models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    input_ids = encoder.encode(prompt)

    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    prompt = sys.argv[1]
    output_text = main(prompt=prompt)
    print(output_text)