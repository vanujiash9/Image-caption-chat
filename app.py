import os
import sys
import torch
import yaml
import webbrowser
import threading
from time import sleep
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, render_template

# === Sửa lỗi đường dẫn và import ===
# Lấy đường dẫn tuyệt đối đến thư mục chứa file app.py (thư mục gốc của dự án)
basedir = os.path.abspath(os.path.dirname(__file__))

# Thêm thư mục 'src' vào đường dẫn tìm kiếm của Python
sys.path.insert(0, os.path.join(basedir, 'src'))

# Import các lớp model bây giờ sẽ hoạt động
from src.model.captioning_model import CaptioningModel

# --- KHỞI TẠO FLASK VỚI ĐƯỜNG DẪN TEMPLATE TƯỜNG MINH ---
app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# --- TẢI MODEL VÀ CÁC THÀNH PHẦN ---
print("--- Initializing server... Please wait. ---")
CONFIG_PATH = os.path.join(basedir, "config/config_main.yaml")

with open(CONFIG_PATH, 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHECKPOINT_PATH = os.path.join(basedir, config['train']['output_dir'], 'best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
model = CaptioningModel(config).to(device)

print(f"--- Loading model from checkpoint: {CHECKPOINT_PATH} ---")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}.")
    
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 
print("--- Model loaded successfully! Server is ready. ---")

transform = transforms.Compose([
    transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['data']['image_mean'], std=config['data']['image_std'])
])

# --- ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_ids = model.generate_caption(
                image_tensor, 
                max_len=config['data']['max_seq_length'], 
                sos_token=tokenizer.bos_token_id, 
                eos_token=tokenizer.eos_token_id, 
                beam_size=config['eval']['beam_size']
            )
        caption = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return jsonify({'caption': caption.strip()})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500

# --- KHỞI CHẠY SERVER ---
def open_browser():
    sleep(1.5)
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000)