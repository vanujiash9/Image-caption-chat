# src/eval_metric/evaluation.py

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

class ScoreCalculator:
    # CẬP NHẬT: Nhận tokenizer để tách từ chính xác
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.smooth = SmoothingFunction().method4
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _compute_bleu(self, references_tokenized, hypotheses_tokenized):
        return corpus_bleu(references_tokenized, hypotheses_tokenized, smoothing_function=self.smooth)

    def _compute_meteor(self, references_tokenized, hypotheses_tokenized):
        total_meteor = 0
        for refs_for_one_hyp, hyp in zip(references_tokenized, hypotheses_tokenized):
            total_meteor += meteor_score(refs_for_one_hyp, hyp)
        return total_meteor / len(hypotheses_tokenized)

    def _compute_rouge_l(self, references_str, hypotheses_str):
        total_rouge = 0
        for ref_group, hyp in zip(references_str, hypotheses_str):
            score = self.rouge.score(ref_group[0], hyp)['rougeL'].fmeasure
            total_rouge += score
        return total_rouge / len(hypotheses_str)

    def compute_all(self, references, hypotheses):
        # CẬP NHẬT: Dùng tokenizer chuyên dụng để tách từ
        refs_tokenized = [
            [self.tokenizer.tokenize(ref) for ref in ref_group] for ref_group in references
        ]
        hyps_tokenized = [self.tokenizer.tokenize(hyp) for hyp in hypotheses]

        bleu_score = self._compute_bleu(refs_tokenized, hyps_tokenized)
        meteor_score_val = self._compute_meteor(refs_tokenized, hyps_tokenized)
        rouge_score = self._compute_rouge_l(references, hypotheses)

        return {
            "BLEU": bleu_score,
            "METEOR": meteor_score_val,
            "ROUGE-L": rouge_score
        }