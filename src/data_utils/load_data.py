# data_utils/load_data.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer # SỬ DỤNG: AutoTokenizer

class MyDataset(Dataset):
    def __init__(self, annotation_path, image_dir, tokenizer, max_seq_length, transform):
        with open(annotation_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        self.annotations = json_data["annotations"]
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transform = transform
        self.imageid2filename = {img['id']: img['filename'] for img in json_data["images"]}
        self.imageid2captions = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.imageid2captions:
                self.imageid2captions[img_id] = []
            self.imageid2captions[img_id].append(ann['caption'])
        self.unique_image_ids = list(self.imageid2captions.keys())

    def __len__(self):
        return len(self.unique_image_ids)

    def __getitem__(self, idx):
        image_id = self.unique_image_ids[idx]
        caption = self.imageid2captions[image_id][0] 
        all_captions = self.imageid2captions[image_id]
        filename = self.imageid2filename.get(image_id)
        if filename is None: raise ValueError(f"Image id {image_id} not found")
        img_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            return None 
        image = self.transform(image)
        tokenized = self.tokenizer(
            " " + caption, # PhoBERT hoạt động tốt hơn khi có dấu cách ở đầu
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized.input_ids.squeeze(0)
        return {"images": image, "captions": input_ids, "image_id": image_id, "all_captions_text": all_captions}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def Get_Loader(config, mode='train'):
    # SỬ DỤNG: AutoTokenizer để tải PhoBERT
    tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
    max_seq_length = config['data']['max_seq_length']
    image_size = config['data']['image_size']
    normalize = transforms.Normalize(mean=config['data']['image_mean'], std=config['data']['image_std'])

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
        annotation_path = config['data']['train_annotation']
        image_dir = config['data']['image_dir']
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
        if mode == 'val':
            annotation_path = config['data']['val_annotation']
            image_dir = config['data']['image_dir'].replace('train-images', 'val-images')
        else:
            annotation_path = config['data']['test_annotation']
            image_dir = config['data']['image_dir'].replace('train-images', 'test-images')
    
    dataset = MyDataset(annotation_path, image_dir, tokenizer, max_seq_length, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'] if mode=='train' else config['eval']['batch_size'],
        shuffle=(mode=='train'),
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader