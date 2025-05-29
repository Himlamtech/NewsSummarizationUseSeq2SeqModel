"""
Evaluation Metrics for Vietnamese Summarization

Implements ROUGE, BLEU, and custom metrics for Vietnamese text evaluation.
"""

import numpy as np
from typing import List, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not available")

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logger.warning("sacrebleu not available")

try:
    from underthesea import word_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False


class VietnameseTextMetrics:
    """Vietnamese-specific text evaluation metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def vietnamese_tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text."""
        if UNDERTHESEA_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        return text.lower().split()
    
    def compute_vietnamese_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores with Vietnamese tokenization."""
        if not ROUGE_AVAILABLE:
            return {'rouge1_fmeasure': 0.0, 'rouge2_fmeasure': 0.0, 'rougeL_fmeasure': 0.0}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            # Tokenize for Vietnamese
            pred_tokens = ' '.join(self.vietnamese_tokenize(pred))
            ref_tokens = ' '.join(self.vietnamese_tokenize(ref))
            
            scores = scorer.score(ref_tokens, pred_tokens)
            
            for metric, score in scores.items():
                rouge_scores[f'{metric}_fmeasure'].append(score.fmeasure)
        
        return {k: np.mean(v) for k, v in rouge_scores.items()}
    
    def compute_vietnamese_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores with Vietnamese tokenization."""
        if not BLEU_AVAILABLE:
            return {'bleu': 0.0}
        
        try:
            bleu = BLEU()
            score = bleu.corpus_score(predictions, [references])
            return {'bleu': score.score / 100.0}
        except:
            return {'bleu': 0.0}
    
    def compute_length_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute length-based metrics."""
        pred_lengths = [len(self.vietnamese_tokenize(pred)) for pred in predictions]
        ref_lengths = [len(self.vietnamese_tokenize(ref)) for ref in references]
        
        if not pred_lengths or not ref_lengths:
            return {}
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        }
    
    def compute_repetition_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute repetition-based metrics."""
        repetition_scores = []
        
        for pred in predictions:
            tokens = self.vietnamese_tokenize(pred)
            if len(tokens) <= 1:
                repetition_scores.append(0.0)
                continue
            
            # Compute bigram repetition
            bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            unique_bigrams = len(set(bigrams))
            
            if len(bigrams) > 0:
                repetition = 1 - (unique_bigrams / len(bigrams))
                repetition_scores.append(repetition)
            else:
                repetition_scores.append(0.0)
        
        return {
            'repetition_score': np.mean(repetition_scores) if repetition_scores else 0.0
        }


class SummarizationEvaluator:
    """Comprehensive evaluator for summarization models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vietnamese_metrics = VietnameseTextMetrics()
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        if len(predictions) != len(references):
            raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
        
        all_metrics = {}
        
        # ROUGE scores
        rouge_scores = self.vietnamese_metrics.compute_vietnamese_rouge(predictions, references)
        all_metrics.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.vietnamese_metrics.compute_vietnamese_bleu(predictions, references)
        all_metrics.update(bleu_scores)
        
        # Length metrics
        length_metrics = self.vietnamese_metrics.compute_length_metrics(predictions, references)
        all_metrics.update(length_metrics)
        
        # Repetition metrics
        repetition_metrics = self.vietnamese_metrics.compute_repetition_metrics(predictions)
        all_metrics.update(repetition_metrics)
        
        # Composite score
        all_metrics['composite_score'] = self._compute_composite_score(all_metrics)
        
        return all_metrics
    
    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute a composite score combining multiple metrics."""
        weights = {
            'rouge1_fmeasure': 0.3,
            'rouge2_fmeasure': 0.3,
            'rougeL_fmeasure': 0.3,
            'bleu': 0.1
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                composite += metrics[metric] * weight
                total_weight += weight
        
        return composite / max(total_weight, 1.0)
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """Print a formatted summary of metrics."""
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        # Core metrics
        print("\n📊 Core Metrics:")
        core_metrics = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure', 'bleu']
        for metric in core_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.4f}")
        
        # Quality metrics
        print("\n🎯 Quality Metrics:")
        quality_metrics = ['repetition_score', 'length_ratio']
        for metric in quality_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.4f}")
        
        # Composite score
        if 'composite_score' in metrics:
            print(f"\n🏆 Composite Score: {metrics['composite_score']:.4f}")
        
        print("="*60)
