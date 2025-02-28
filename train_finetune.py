import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
from torch.utils.data import Dataset, DataLoader
from constants import *
from tqdm import tqdm

from dataset_finetune import get_vocab, get_dataloader

from model import MiniLLaDA


if __name__ == "__main__":

    dataloader = get_dataloader()
    vocab, pad_token_id, mask_token_id, sep_token_id = get_vocab()
    vocab_size = len(vocab)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = MiniLLaDA(vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, max_len=MAX_LEN).to(device)
    model.load_state_dict(torch.load("mini_llada.pth"))
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=LR)


    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader):
            batch = batch.to(device)  # Shape: (batch_size, seq_len)
            optimizer.zero_grad()
            batch_size, seq_len = batch.size()
            
            # Save original tokens as targets
            targets = batch.clone()
            # Create a boolean mask for positions that will be masked
            mask_indicator = torch.zeros_like(batch, dtype=torch.bool)
            
            # Apply masking only to tokens after the separator.
            for i in range(batch_size):
                # Find the first occurrence of the SEP token in the sample.
                sep_positions = (batch[i] == sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) > 0:
                    sep_index = sep_positions[0].item()
                else:
                    # If for some reason SEP_TOKEN is not found, consider the entire sample as response.
                    sep_index = -1  
                # The response portion starts after sep_index.
                # We mask tokens only for indices > sep_index.
                response_range = range(sep_index + 1, seq_len)
                t = random.uniform(0, 1)
                # Generate random values for only the response portion.
                rand_vals = torch.rand(len(response_range), device=device)
                sample_mask = (rand_vals < t)
                for j, idx in enumerate(response_range):
                    # Only mask non-padding tokens.
                    if sample_mask[j] and batch[i, idx] != pad_token_id:
                        mask_indicator[i, idx] = True
                        batch[i, idx] = mask_token_id
            
            # Forward pass
            logits = model(batch)  # (batch_size, seq_len, vocab_size)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            mask_indicator = mask_indicator.view(-1)
            
            # Compute cross-entropy loss only for masked positions
            loss_all = criterion(logits, targets)
            if mask_indicator.float().sum() > 0:
                loss = (loss_all * mask_indicator.float()).sum() / mask_indicator.float().sum()
            else:
                loss = loss_all.mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "mini_llada.pth")