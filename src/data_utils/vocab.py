from typing import Dict, List
import json
import re

def tokenize(self, text: str) -> List[str]:
    text = text.lower()
    # Loại bỏ dấu câu, giữ lại chữ và số
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    return tokens

class Vocabulary:
    def __init__(self, captions: List[str] = None, special_tokens: List[str] = None):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '<sos>', '<eos>']
        if captions:
            self.build_vocab(captions)
    
    def build_vocab(self, captions: List[str]):
        word_counts = {}
        for caption in captions:
            for word in caption.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        vocab = self.special_tokens + sorted(word_counts.keys(), key=lambda w: -word_counts[w])
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def __getitem__(self, token):
        return self.word_to_idx.get(token, self.word_to_idx['[UNK]'])
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Loại bỏ dấu câu, giữ lại chữ và số
        tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word_to_idx.get(token, self.word_to_idx['[UNK]']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.idx_to_word.get(i, '[UNK]') for i in ids]
    
    def vocab_size(self) -> int:
        return len(self.word_to_idx)
    
    def pad_token_id(self) -> int:
        return self.word_to_idx['[PAD]']
    
    def save_to_file(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.special_tokens = data['special_tokens']