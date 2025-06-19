import os
import sys
import torch
import yaml
import argparse
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
# === SỬA LỖI IMPORT: Thêm thư mục 'src' vào đường dẫn tìm kiếm của Python ===
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)
# =========================================================================

# Import các lớp model từ dự án của bạn
from src.model.captioning_model import CaptioningModel

def predict_caption(image_path, config_path):
    """Tải model đã train và dự đoán chú thích cho một ảnh duy nhất."""
    # Tải cấu hình với encoding utf-8
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Lấy đường dẫn checkpoint từ config
    checkpoint_path = os.path.join(config['train']['output_dir'], 'best_model.pth')

    # Thiết lập
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
    
    # Tải Model và trọng số
    model = CaptioningModel(config).to(device)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['image_mean'], std=config['data']['image_std'])
    ])

    # Tải và xử lý ảnh
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán
    print("Generating caption...")
    with torch.no_grad():
        generated_ids = model.generate_caption(
            image_tensor,
            max_len=config['data']['max_seq_length'],
            sos_token=tokenizer.bos_token_id,
            eos_token=tokenizer.eos_token_id,
            beam_size=config['eval']['beam_size']
        )

    # In kết quả
    generated_caption = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print("\n" + "="*50)
    print(f"Image: {image_path}")
    print(f"AI Caption: {generated_caption.strip()}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a caption for a single image.")
    parser.add_argument('--image_path', required=True, help="Path to the input image.")
    parser.add_argument('--config', default="config/config_main.yaml", help="Path to the config YAML file.")
    args = parser.parse_args()
    predict_caption(args.image_path, args.config)