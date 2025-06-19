import json

files = [
    "data/train/train_data.json",
    "data/val/val_data.json", 
    "data/test/test_data.json"
]

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:  # Fix encoding
            data = json.load(f)
        
        if isinstance(data, dict) and 'annotations' in data:
            count = len(data['annotations'])
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0
            
        print(f"{file}: {count} annotations")
        
    except Exception as e:
        print(f"{file}: ERROR - {e}")

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