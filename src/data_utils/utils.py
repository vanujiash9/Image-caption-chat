import torch
import re
import json
import unicodedata
from typing import List

# ========== XỬ LÝ TEXT TIẾNG VIỆT ==========

def normalize_vietnamese(text: str) -> str:
    """Normalize tiếng Việt với chuẩn NFC và loại ký tự lạ"""
    if not text:
        return ""
    text = text.strip().lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u00C0-\u1EF9.,!?-]', '', text)
    return text.strip()

def preprocess_vietnamese_caption(caption: str) -> str:
    """Làm sạch caption tiếng Việt, đảm bảo có nội dung"""
    caption = normalize_vietnamese(caption)
    return caption if len(caption) >= 3 else "có một hình ảnh"

def remove_punctuation(text: str) -> str:
    """Xoá hết dấu câu, dùng cho phân tích hoặc đánh giá"""
    text = text.replace('\n', ' ').strip().lower()
    text = re.sub(r'[\"“”‘’,.:;!?()\-\']', '', text)
    text = " ".join(text.split())
    return text

# ========== THỐNG KÊ MODEL ==========

def countTrainableParameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def countParameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

# ========== HỖ TRỢ ĐỌC DỮ LIỆU JSON ==========

def load_captions_from_json(json_path: str, use_segment: bool = True) -> List[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    key = 'segment_caption' if use_segment else 'caption'
    return [entry[key] for entry in data if key in entry]
