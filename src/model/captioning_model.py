# src/model/captioning_model.py

import torch
import torch.nn as nn
# Dòng import này không còn cần thiết nếu ViCapBERT đã được sửa
# from model.vi_caption_model import ViCapBERT

class CaptioningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # device sẽ được xử lý bởi Trainer, model sẽ được chuyển sang device sau khi khởi tạo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cần import ViCapBERT ở đây
        from model.vi_caption_model import ViCapBERT
        self.model = ViCapBERT(config)

        # Các thuộc tính này không cần thiết ở lớp Wrapper này
        # vocab_size = self.model.vocab_size
        # pad_token_id = config['data'].get('pad_token_id', self.model.pad_token_id)
        # self.loss_fn = ...

    def forward(self, images, captions=None):
        # Hàm forward chỉ cần đơn giản gọi đến model bên trong
        # Việc chuyển sang device đã được ViCapBERT xử lý
        if captions is not None:
            logits = self.model(images, captions)
            # Trả về logits và None để tương thích với cấu trúc của Trainer
            return logits, None
        else:
            # Khi không có caption, gọi chế độ generate
            # Tham số sẽ được truyền từ Trainer
            # Đây là một điểm yếu trong thiết kế, nên tách bạch train và generate
            # Nhưng để giữ nguyên cấu trúc, ta sẽ để trống ở đây
            raise NotImplementedError("Generate mode should be called via generate_caption method.")

    # Hàm này nên được đặt trong Trainer, nhưng để sửa lỗi, ta sẽ sửa tại đây
    def generate_caption(self, images, max_len=20, sos_token=None, eos_token=None, beam_size=1):
        self.model.eval()
        with torch.no_grad():
            return self.model.generate_caption(
                images,
                max_len=max_len,
                sos_token=sos_token,
                eos_token=eos_token,
                beam_size=beam_size
            )
            
    # Chuyển model và các submodule của nó sang đúng device
    def to(self, device):
        self.device = device
        self.model.to(device)
        return super().to(device)