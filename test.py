import torch
import torch.nn as nn
import numpy as np
import math
from constants import *
from model import MiniLLaDA
from dataset import get_vocab, get_tokenizer
import random

vocab, pad_token_id, mask_token_id = get_vocab()
vocab_size = len(vocab)
# id2word = {i: w for w, i in vocab.items()}

hf_tokenizer = get_tokenizer()

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

model = MiniLLaDA(vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, max_len=MAX_LEN).to(device)


def generate_text(prompt, response_length=64, num_steps=16, temperature=1.0):
    """
    Generate text following LLaDA's reverse process logic:
      - We define time steps t in descending order: t = 1, 1 - 1/N, ..., 0
      - At each step, we compute r'_t = argmax pθ(r0 | p0, r_t)
      - If r'_t[i] == [MASK], keep the old token; otherwise, with probability (s/t) re-mask it, else keep r'_t[i].
      - s = t - 1/N

    Args:
      prompt (str): The user prompt.
      response_length (int): Number of tokens in the generated response.
      num_steps (int): Number of time steps (N) in the reverse process.
      temperature (float): Temperature for scaling logits (optional, default=1.0).

    Returns:
      str: The concatenation of the prompt and the generated response, decoded via hf_tokenizer.
    """
    model.eval()

    prompt_enc = hf_tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_enc['input_ids']
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Create a fully masked response tensor
    response_tensor = torch.full((1, response_length), mask_token_id, dtype=torch.long).to(device)

    # Concatenate prompt + masked response
    full_seq = torch.cat([prompt_tensor, response_tensor], dim=1)
    prompt_len = prompt_tensor.size(1)

    for step in range(num_steps):
        # Current time t and step s as in the paper
        t = 1.0 - (step / num_steps)
        s = 1.0 - ((step + 1) / num_steps)
        if s < 0:
            s = 0

        # Forward pass
        logits = model(full_seq)  # shape: (1, total_len, vocab_size)
        response_logits = logits[:, prompt_len:, :]  # shape: (1, response_length, vocab_size)

        # Optional: temperature scaling
        if temperature != 1.0:
            response_logits = response_logits / temperature

        # greedy, take the token with the highest probability
        r_prime = torch.argmax(response_logits, dim=-1)  # shape: (1, response_length)

        # Construct new response
        old_response = full_seq[:, prompt_len:].clone()
        new_response = old_response.clone()

        for i in range(response_length):
            predicted_token = r_prime[0, i].item()
            # If the model still predicts [MASK], keep the old token
            if predicted_token == mask_token_id:
                new_response[0, i] = old_response[0, i]
            else:
                # Probability to re-mask is s/t (assuming t>0)
                p_remask = (s / t) if t > 0 else 0.0
                if random.random() < p_remask:
                    new_response[0, i] = mask_token_id
                else:
                    new_response[0, i] = predicted_token

        full_seq[:, prompt_len:] = new_response

    # decoding
    final_ids = full_seq.squeeze().tolist()
    # skip_special_tokens=True will remove tokens like [CLS], [SEP], etc.
    decoded_text = hf_tokenizer.decode(final_ids, skip_special_tokens=True)

    return decoded_text

if __name__ == "__main__":

    print(generate_text("User: Hey! <sep> Assistant: ", response_length=15, num_steps=3))
