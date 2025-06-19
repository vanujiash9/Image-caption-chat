# src/model/vi_caption_model.py

import torch
import torch.nn as nn
from transformers import AutoModel
import timm
import torch.nn.functional as F

class ViCapBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === SỬA ĐỔI 1: Tải Swin Transformer và không thay đổi cấu trúc của nó ===
        self.vision_encoder = timm.create_model(
            config['model']['vit_name'],
            pretrained=True,
            num_classes=0  # Thiết lập num_classes=0 để loại bỏ lớp head (classifier)
        )
        
        # Lấy vision_dim một cách an toàn
        vision_dim = self.vision_encoder.num_features

        # Text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(config['model']['bert_model'])
        bert_dim = self.text_encoder.config.hidden_size

        self.encoder_proj = nn.Linear(vision_dim, bert_dim)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=bert_dim, nhead=8, dim_feedforward=bert_dim * 4),
            num_layers=config['model']['num_decoder_layers']
        )

        self.vocab_size = self.text_encoder.config.vocab_size
        self.output_layer = nn.Linear(bert_dim, self.vocab_size)
        self.pad_token_id = self.text_encoder.config.pad_token_id

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

    def forward(self, images, captions):
        images = images.to(self.device)
        captions = captions.to(self.device)

        # === SỬA ĐỔI 2: Lấy đặc trưng trực tiếp ===
        # Đầu ra của Swin Transformer (với num_classes=0) sẽ là một vector đặc trưng
        # có kích thước (Batch, num_features). Ví dụ: (24, 1024)
        vision_feats = self.vision_encoder(images) 

        # Bây giờ vision_feats đã có kích thước đúng (24, 1024) để đưa vào encoder_proj
        # Kích thước của memory sẽ là (1, Batch, bert_dim), ví dụ (1, 24, 768)
        memory = self.encoder_proj(vision_feats).unsqueeze(0)

        tgt_input = captions
        
        tgt_emb = self.text_encoder.embeddings(tgt_input).transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))
        tgt_padding_mask = (tgt_input == self.pad_token_id)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.output_layer(decoded.transpose(0, 1))
        return output

    # Sửa lại hàm generate_caption cho nhất quán
    def generate_caption(self, images, max_len=40, sos_token=0, eos_token=2, beam_size=3):
        self.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            images = images.to(self.device)
            vision_feats = self.vision_encoder(images)
            memory = self.encoder_proj(vision_feats).unsqueeze(0)

            if beam_size == 1:
                # ... (giữ nguyên logic beam_size=1)
                captions = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=self.device)
                for _ in range(max_len - 1):
                    tgt_emb = self.text_encoder.embeddings(captions).transpose(0, 1)
                    tgt_mask = self.generate_square_subsequent_mask(captions.size(1))
                    out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = self.output_layer(out[-1])
                    next_tokens = logits.argmax(dim=-1).unsqueeze(1)
                    captions = torch.cat([captions, next_tokens], dim=1)
                    if (next_tokens == eos_token).all(): break
                return captions
            else:
                # ... (giữ nguyên logic beam_size > 1)
                assert batch_size == 1, "Beam search hiện tại chỉ hỗ trợ batch_size=1."
                beam = [(torch.tensor([sos_token], device=self.device), 0.0)]
                for _ in range(max_len - 1):
                    new_beam = []
                    for seq, score in beam:
                        if seq[-1].item() == eos_token:
                            new_beam.append((seq, score))
                            continue
                        tgt_emb = self.text_encoder.embeddings(seq.unsqueeze(0)).transpose(0, 1)
                        tgt_mask = self.generate_square_subsequent_mask(seq.size(0))
                        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                        logits = self.output_layer(out[-1])
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                        for k in range(beam_size):
                            next_token = topk_indices[k].unsqueeze(0)
                            new_seq = torch.cat([seq, next_token])
                            new_score = score + topk_log_probs[k].item()
                            new_beam.append((new_seq, new_score))
                    new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
                    beam = new_beam
                    if all(s[-1].item() == eos_token for s, _ in beam): break
                return beam[0][0].unsqueeze(0)