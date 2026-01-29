import torch
import tiktoken
import random
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

class TokenProvider:
    def __init__(self, batch_size=4, seq_len=128, device='cuda'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def get_batch(self):
        raise NotImplementedError

# ---------------------------------------------------------------------------------------------------------------------------

class RandomTokenProvider(TokenProvider):
    def __init__(self, vocab_size=50257, batch_size=4, seq_len=128, device='cuda'):
        super().__init__(batch_size, seq_len, device)
        self.vocab_size = vocab_size

    def get_batch(self):
        return torch.randint(
            0, self.vocab_size, 
            (self.batch_size, self.seq_len), 
            device=self.device
        )

# ---------------------------------------------------------------------------------------------------------------------------

class TokenFromTextProvider(TokenProvider):
    def __init__(self, dataset, tokenizer_name="gpt2", batch_size=4, seq_len=128, device='cuda', 
                 shuffle_tokens=False, buffer_size=10000, text_column_name=None):
        super().__init__(batch_size, seq_len, device)
        
        self.dataset = dataset
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.shuffle_tokens = shuffle_tokens
        self.target_buffer_size = buffer_size if shuffle_tokens else (batch_size * seq_len * 2)
        
        self.dataset_iter = iter(self.dataset)
        self.token_buffer = []
        
        self.text_column = text_column_name # Can be None initially
        self.candidates = ['text', 'content', 'sentence', 'document', 'context', 'question', 'body']

    def _get_text_from_item(self, item):
        if isinstance(item, str):
            return item
            
        if isinstance(item, dict):
            if self.text_column is None:
                keys = item.keys()
                for candidate in self.candidates:
                    if candidate in keys:
                        self.text_column = candidate
                        print(f"TokenProvider: Auto-detected text column: '{self.text_column}'")
                        break
                
                if self.text_column is None:
                    for k, v in item.items():
                        if isinstance(v, str) and len(v) > 0:
                            self.text_column = k
                            print(f"TokenProvider: Auto-detected text column: '{self.text_column}' (heuristic)")
                            break
            
            if self.text_column and self.text_column in item:
                return item[self.text_column]
            else:
                raise ValueError(f"Could not find text column in dataset item. Available keys: {list(item.keys())}")

        raise TypeError(f"Dataset item must be dict or str, got {type(item)}")

    def _fill_buffer(self):
        required_tokens = self.batch_size * self.seq_len
        
        while len(self.token_buffer) < max(required_tokens, self.target_buffer_size):
            try:
                item = next(self.dataset_iter)
                text = self._get_text_from_item(item)
                if text:
                    new_ids = self.enc.encode(text, allowed_special={'<|endoftext|>'})
                    self.token_buffer.extend(new_ids)
                    
            except StopIteration:
                self.dataset_iter = iter(self.dataset)

    def get_batch(self):
        batch_len = self.batch_size * self.seq_len
        if len(self.token_buffer) < batch_len:
            self._fill_buffer()
        if self.shuffle_tokens:
            random.shuffle(self.token_buffer)
        
        batch_tokens = self.token_buffer[:batch_len]
        self.token_buffer = self.token_buffer[batch_len:]
        return torch.tensor(batch_tokens, device=self.device).view(self.batch_size, self.seq_len)

    def reset(self):
        self.token_buffer = []
        if hasattr(self, 'dataset'):
            self.dataset_iter = iter(self.dataset)

    def close(self):
        if hasattr(self, 'dataset_iter'):
            del self.dataset_iter
            self.dataset_iter = None

# ---------------------------------------------------------------------------------------------------------------------------

DATASET_CONFIGS = {
    "TinyShakespeare": ("Trelis/tiny-shakespeare", None, "Text"),
    "OpenWebText": ("Skylion007/openwebtext", None, "text"),
    "FineWeb-Edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "RefinedWeb": ("tiiuae/falcon-refinedweb", None, "content"),
    "Cosmopedia": ("HuggingFaceTB/cosmopedia", "stories", "text"),
    "CodeParrot": ("codeparrot/codeparrot-clean", None, "content"),
    "ArXiv Science": ("ccdv/arxiv-summarization", None, "abstract"),
    "Orca Math": ("microsoft/orca-math-word-problems-200k", None, "question"),
    "GSM8k Math": ("openai/gsm8k", "main", "question"),
    "Alpaca Cleaned": ("yahma/alpaca-cleaned", None, "output"),
    "Medical PubMed": ("medalpaca/medical_meadow_pubmed_causal", None, "input"),
    "Yelp Reviews": ("yelp_review_full", None, "text"),
}

def stream_dataset_from_config(dataset_name, split="train"):
    hf_id, subset, text_column_name = DATASET_CONFIGS[dataset_name]
    if subset is None:
        ds = load_dataset(hf_id, split=split, streaming=True) 
    else:
        ds = load_dataset(hf_id, subset, split=split, streaming=True)

    return ds, text_column_name
