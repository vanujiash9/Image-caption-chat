# task/infer.py

import os
import torch
from tqdm import tqdm
# SỬA: Dùng AutoTokenizer
from transformers import AutoTokenizer 
from model.captioning_model import CaptioningModel
from data_utils.load_data import Get_Loader
from eval_metric.evaluation import ScoreCalculator
import json

class Inference:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # SỬA: Load đúng tokenizer và truyền vào ScoreCalculator
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
        self.scorer = ScoreCalculator(tokenizer=self.tokenizer)

        # Load test data
        # CHÚ Ý: Cần truyền tokenizer vào Get_Loader để nó hoạt động đúng
        self.test_loader = Get_Loader(config, mode='test')

        # Init model & load weights
        self.model = CaptioningModel(config).to(self.device)
        best_ckpt_path = os.path.join(config['train']['output_dir'], 'best_model.pth')
        
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best model checkpoint not found at: {best_ckpt_path}")

        print(f"--- Loading best model from: {best_ckpt_path} ---")
        checkpoint = torch.load(best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Output
        self.output_path = os.path.join(config['train']['output_dir'], 'predictions.json')

    def generate_captions(self):
        all_references = []
        all_hypotheses = []
        predictions = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Generating Captions"):
                if batch is None: continue

                images = batch['images'].to(self.device)
                # Lấy tất cả các caption tham chiếu từ batch
                all_captions_text = batch['all_captions_text'] 

                generated_ids = self.model.generate_caption(
                    images,
                    max_len=self.config['data']['max_seq_length'],
                    sos_token=self.tokenizer.bos_token_id,
                    eos_token=self.tokenizer.eos_token_id,
                    beam_size=self.config['eval']['beam_size']
                )

                pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Sửa lại logic để xử lý đúng batch_size=1
                for i in range(len(pred_texts)):
                    refs_for_one_image = [all_captions_text[j][i] for j in range(len(all_captions_text))]
                    all_references.append(refs_for_one_image)
                    all_hypotheses.append(pred_texts[i].strip())

                    predictions.append({
                        'image_id': batch['image_id'][i].item(),
                        'references': refs_for_one_image,
                        'generated': pred_texts[i].strip()
                    })

        # Save predictions
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        print(f"Predictions saved to {self.output_path}")

        # Evaluate
        print("\n📊 Final Evaluation on Test Set:")
        scores = self.scorer.compute_all(all_references, all_hypotheses)
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")