# Tạo file debug_tokens.py
from transformers import AutoTokenizer
import torch

# Check tokenizer vocab size
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Config vocab size: 30522")

# Test tokenization
test_caption = "người phụ nữ đang cõng một đứa trẻ trên lưng"
tokens = tokenizer(test_caption, return_tensors='pt', padding='max_length',
                   truncation=True, max_length=40)

input_ids = tokens['input_ids'].squeeze(0)
print(f"Token IDs: {input_ids}")
print(f"Max token ID: {input_ids.max().item()}")
print(f"Min token ID: {input_ids.min().item()}")

# Check if any token ID >= vocab_size
invalid_tokens = input_ids >= 30522
if invalid_tokens.any():
    print(f"❌ Found invalid tokens: {input_ids[invalid_tokens]}")
else:
    print("✅ All tokens valid")