# task/train.py

import os
import time
import torch
import torch.nn as nn
# TỐI ƯU HÓA: Import các công cụ training hiện đại
# Đoạn code ĐÃ SỬA
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW # <--- LẤY AdamW TỪ torch.optim
from tqdm import tqdm
from eval_metric.evaluation import ScoreCalculator
from model.captioning_model import CaptioningModel
from data_utils.load_data import Get_Loader

class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.config = config

        self.train_loader = Get_Loader(config, mode='train')
        self.val_loader = Get_Loader(config, mode='val')

        # NÂNG CẤP: Dùng AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
        # CẬP NHẬT: Token đặc biệt của PhoBERT
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token = self.tokenizer.bos_token_id # bos_token_id
        self.eos_token = self.tokenizer.eos_token_id # eos_token_id

        self.model = CaptioningModel(config).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,
            label_smoothing=config['train'].get('label_smoothing', 0.0)
        )

        # NÂNG CẤP: Dùng AdamW Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config['train']['learning_rate']),
            weight_decay=float(config['train']['weight_decay'])
        )

        # NÂNG CẤP: Dùng Linear Warmup + Cosine Decay Scheduler
        num_training_steps = len(self.train_loader) * config['train']['num_train_epochs']
        num_warmup_steps = int(config['train']['warmup_steps'] * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # NÂNG CẤP: Truyền tokenizer vào ScoreCalculator
        self.scorer = ScoreCalculator(tokenizer=self.tokenizer)
        
        # Giữ nguyên các tham số training khác
        self.max_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.metric = config['train']['metric_for_best_model']
        self.save_path = config['train']['output_dir']
        self.beam_size = config['eval'].get('beam_size', 1)
        os.makedirs(self.save_path, exist_ok=True)
        self.best_score = 0
        self.start_epoch = 1

        # Giữ lại khả năng resume training
        last_ckpt_path = os.path.join(self.save_path, 'last_model.pth')
        if os.path.exists(last_ckpt_path):
            print(f"--- Found last checkpoint. Resuming training from {last_ckpt_path} ---")
            checkpoint = torch.load(last_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Tải lại cả scheduler
            self.start_epoch = checkpoint['epoch'] + 1
            best_ckpt_path = os.path.join(self.save_path, 'best_model.pth')
            if os.path.exists(best_ckpt_path):
                self.best_score = torch.load(best_ckpt_path, map_location=self.device)['score']
                print(f"Resuming with best score so far ({self.metric}): {self.best_score:.4f}")

    def train(self):
        early_stop_count = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} - Training"):
                if batch is None: continue
                images, captions = batch['images'].to(self.device), batch['captions'].to(self.device)
                input_captions, target_captions = captions[:, :-1], captions[:, 1:]
                self.optimizer.zero_grad()
                logits = self.model.model(images, input_captions) # Gọi thẳng vào model bên trong
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() # Cập nhật scheduler sau mỗi batch
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            val_bleu, val_rouge, val_loss = self.validate()
            score = val_bleu if self.metric == 'BLEU' else val_rouge
            self.log_epoch(epoch, avg_train_loss, val_bleu, val_rouge, val_loss, time.time() - start_time)

            # Lưu checkpoint bao gồm cả scheduler
            current_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'score': score
            }
            torch.save(current_checkpoint, os.path.join(self.save_path, 'last_model.pth'))
            if score > self.best_score:
                self.best_score = score
                torch.save(current_checkpoint, os.path.join(self.save_path, 'best_model.pth'))
                print(f"** New best score ({self.metric}): {score:.4f}. Saving model... **")
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    def validate(self):
        self.model.eval()
        all_references, all_hypotheses, total_loss = [], [], 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if batch is None: continue
                images, captions = batch['images'].to(self.device), batch['captions'].to(self.device)
                all_captions_text = batch['all_captions_text']
                input_captions, target_captions = captions[:, :-1], captions[:, 1:]
                logits = self.model.model(images, input_captions)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1))
                total_loss += loss.item()
                generated_ids = self.model.generate_caption(
                    images, max_len=self.config['data']['max_seq_length'],
                    sos_token=self.sos_token, eos_token=self.eos_token, beam_size=self.beam_size
                )
                pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for i in range(len(pred_texts)):
                    all_references.append([all_captions_text[j][i] for j in range(len(all_captions_text))])
                    all_hypotheses.append(pred_texts[i].strip())
        scores = self.scorer.compute_all(all_references, all_hypotheses)
        return scores["BLEU"], scores["ROUGE-L"], total_loss / len(self.val_loader)

    def log_epoch(self, epoch, train_loss, val_bleu, val_rouge, val_loss, elapsed_time):
        print("=" * 60)
        print(f"Epoch {epoch} | Time: {elapsed_time:.2f}s | Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val BLEU: {val_bleu:.4f} | Val ROUGE-L: {val_rouge:.4f}")
        print("=" * 60)
