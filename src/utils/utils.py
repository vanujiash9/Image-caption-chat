import torch
import re
import unicodedata
from typing import List

def countTrainableParameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def countParameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def preprocess_vietnamese_caption(caption: str) -> str:
    if not caption:
        return ""
    
    caption = caption.strip().lower()
    caption = unicodedata.normalize('NFC', caption)
    caption = re.sub(r'\s+', ' ', caption)
    caption = re.sub(r'[^\w\s\u00C0-\u1EF9.,!?-]', '', caption)
    
    return caption.strip()

def clean_vietnamese_text(text: str) -> str:
    text = preprocess_vietnamese_caption(text)
    if len(text) < 3:
        return "có một hình ảnh"
    return text