import glob
import polars as pl
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from constants import BATCH_SIZE, MAX_LEN, ROWS
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
from transformers import GPT2Tokenizer

# Download NLTK punkt data if not already downloaded.
nltk.download('punkt')

# Global variable for worker tokenizer.
worker_tokenizer = None

def init_worker():
    global worker_tokenizer
    worker_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    nltk.download('punkt', quiet=True)


def preprocess_text(text, max_tokens=MAX_LEN):
    """
    Splits a long text (e.g. an entire web page) into segments no longer than max_tokens.
    Uses NLTK to split into sentences and the local worker_tokenizer for tokenization.
    """
    global worker_tokenizer
    sentences = nltk.tokenize.sent_tokenize(text)
    segments = []
    current_segment = ""
    for sentence in sentences:
        candidate = current_segment + " " + sentence if current_segment else sentence
        tokenized = worker_tokenizer.tokenize(candidate)
        if len(tokenized) <= max_tokens:
            current_segment = candidate
        else:
            if current_segment:
                segments.append(current_segment.strip())
            tokenized_sentence = worker_tokenizer.tokenize(sentence)
            if len(tokenized_sentence) > max_tokens:
                for i in range(0, len(tokenized_sentence), max_tokens):
                    chunk_tokens = tokenized_sentence[i:i+max_tokens]
                    chunk_text = worker_tokenizer.convert_tokens_to_string(chunk_tokens)
                    segments.append(chunk_text)
                current_segment = ""
            else:
                current_segment = sentence
    if current_segment:
        segments.append(current_segment.strip())
    return segments


def load_refinedweb_dataset(path_pattern="data/train-*.parquet"):
    files = glob.glob(path_pattern)
    texts = []
    for file in files:
        df = pl.read_parquet(file)
        col_name = df.columns[0]
        texts.extend(df[col_name].to_list())
    print(f"Loaded {len(texts)} raw texts from RefinedWeb dataset")
    return random.sample(texts, ROWS)

def process_all_texts(raw_texts, max_tokens=MAX_LEN):
    all_segments = []
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as executor:
        futures = [executor.submit(preprocess_text, text, max_tokens) for text in raw_texts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing texts"):
            try:
                segments = future.result()
                all_segments.extend(segments)
            except Exception as e:
                print("Error during preprocessing:", e)
    return all_segments

class TextDataset(Dataset):
    def __init__(self, segments, tokenizer):
        self.data = []
        for seg in segments:
            encoded_dict = tokenizer(
                seg,
                add_special_tokens=False,
                max_length=MAX_LEN,
                truncation=True
            )
            input_ids = encoded_dict['input_ids']
            self.data.append(torch.tensor(input_ids, dtype=torch.long))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_token_id):
    batch_size = len(batch)
    lengths = [len(seq) for seq in batch]
    max_len_in_batch = max(lengths)
    padded = torch.full((batch_size, max_len_in_batch), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


def get_vocab():
    hf_tokenizer = get_tokenizer()
    vocab = hf_tokenizer.get_vocab()
    return vocab, hf_tokenizer.pad_token_id, hf_tokenizer.mask_token_id

def get_tokenizer():
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if gpt2_tokenizer.mask_token is None:
        gpt2_tokenizer.add_special_tokens({"mask_token": "<mask>"})
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if gpt2_tokenizer.sep_token is None:
        gpt2_tokenizer.add_special_tokens({"sep_token": "<sep>"})
    return gpt2_tokenizer

def get_dataloader():
    raw_texts = load_refinedweb_dataset()
    all_segments = process_all_texts(raw_texts, MAX_LEN)
    print(f"Total segments after preprocessing: {len(all_segments)}")
    hf_tokenizer = get_tokenizer()
    dataset = TextDataset(all_segments, hf_tokenizer)
    pad_token_id = hf_tokenizer.pad_token_id
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token_id))

if __name__ == '__main__':
    raw_texts = load_refinedweb_dataset()
    all_segments = process_all_texts(raw_texts, MAX_LEN)
    print(f"Total segments after preprocessing: {len(all_segments)}")
    
    hf_tokenizer = get_tokenizer()
    
    dataset = TextDataset(all_segments, hf_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    vocab, pad_token_id, mask_token_id = get_vocab()
    vocab_size = len(vocab)
    print("Vocab size:", vocab_size)
    print("Mask token ID:", mask_token_id)
    print("Pad token ID:", pad_token_id)