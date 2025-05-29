"""
Comprehensive Evaluation Metrics for Vietnamese Summarization
Implements ROUGE, BLEU, BERTScore, and custom Vietnamese-specific metrics
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import re

# Evaluation libraries
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from bert_score import score as bert_score
import evaluate

# Vietnamese NLP
from underthesea import word_tokenize
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class VietnameseTextMetrics:
    """Vietnamese-specific text evaluation metrics"""
    
    def __init__(self):
        self.setup_logging()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def vietnamese_tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using underthesea"""
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            self.logger.warning(f"Vietnamese tokenization failed: {e}")
            return text.lower().split()
    
    def compute_vietnamese_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores with Vietnamese tokenization"""
        smoothing = SmoothingFunction().method1
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.vietnamese_tokenize(pred)
            ref_tokens = [self.vietnamese_tokenize(ref)]
            
            # Compute BLEU-n scores
            for n in range(1, 5):
                weights = [1.0/n] * n + [0.0] * (4-n)
                try:
                    bleu_n = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothing)
                    bleu_scores[f'bleu_{n}'].append(bleu_n)
                except:
                    bleu_scores[f'bleu_{n}'].append(0.0)
        
        # Average scores
        return {k: np.mean(v) for k, v in bleu_scores.items()}
    
    def compute_vietnamese_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores with Vietnamese tokenization"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            # Tokenize for Vietnamese
            pred_tokens = ' '.join(self.vietnamese_tokenize(pred))
            ref_tokens = ' '.join(self.vietnamese_tokenize(ref))
            
            scores = scorer.score(ref_tokens, pred_tokens)
            
            for metric, score in scores.items():
                rouge_scores[f'{metric}_precision'].append(score.precision)
                rouge_scores[f'{metric}_recall'].append(score.recall)
                rouge_scores[f'{metric}_fmeasure'].append(score.fmeasure)
        
        # Average scores
        return {k: np.mean(v) for k, v in rouge_scores.items()}
    
    def compute_length_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute length-based metrics"""
        pred_lengths = [len(self.vietnamese_tokenize(pred)) for pred in predictions]
        ref_lengths = [len(self.vietnamese_tokenize(ref)) for ref in references]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
            'length_diff': np.mean([abs(p - r) for p, r in zip(pred_lengths, ref_lengths)])
        }
    
    def compute_repetition_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute repetition-based metrics"""
        repetition_scores = []
        
        for pred in predictions:
            tokens = self.vietnamese_tokenize(pred)
            if len(tokens) == 0:
                repetition_scores.append(0.0)
                continue
            
            # Compute n-gram repetition
            bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
            
            # Repetition ratio
            unique_bigrams = len(set(bigrams))
            unique_trigrams = len(set(trigrams))
            
            bigram_rep = 1 - (unique_bigrams / max(len(bigrams), 1))
            trigram_rep = 1 - (unique_trigrams / max(len(trigrams), 1))
            
            repetition_scores.append((bigram_rep + trigram_rep) / 2)
        
        return {
            'repetition_score': np.mean(repetition_scores),
            'low_repetition_ratio': sum(1 for score in repetition_scores if score < 0.1) / len(repetition_scores)
        }


class SummarizationEvaluator:
    """Comprehensive evaluator for summarization models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vietnamese_metrics = VietnameseTextMetrics()
        self.setup_logging()
        
        # Initialize evaluation metrics
        self.metrics = config.get('evaluation', {}).get('metrics', ['rouge', 'bleu', 'bertscore'])
        
        # Initialize BERTScore model
        if 'bertscore' in self.metrics:
            self.bert_scorer = evaluate.load("bertscore")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        return self.vietnamese_metrics.compute_vietnamese_rouge(predictions, references)
    
    def compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores"""
        # Vietnamese BLEU
        vietnamese_bleu = self.vietnamese_metrics.compute_vietnamese_bleu(predictions, references)
        
        # Standard BLEU using sacrebleu
        try:
            bleu = BLEU()
            sacrebleu_score = bleu.corpus_score(predictions, [references])
            vietnamese_bleu['sacrebleu'] = sacrebleu_score.score / 100.0
        except Exception as e:
            self.logger.warning(f"SacreBLEU computation failed: {e}")
            vietnamese_bleu['sacrebleu'] = 0.0
        
        return vietnamese_bleu
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore"""
        try:
            # Use multilingual BERT for Vietnamese
            results = self.bert_scorer.compute(
                predictions=predictions,
                references=references,
                model_type="bert-base-multilingual-cased",
                lang="vi"
            )
            
            return {
                'bertscore_precision': np.mean(results['precision']),
                'bertscore_recall': np.mean(results['recall']),
                'bertscore_f1': np.mean(results['f1'])
            }
        except Exception as e:
            self.logger.warning(f"BERTScore computation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }
    
    def compute_meteor_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute METEOR scores"""
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.vietnamese_metrics.vietnamese_tokenize(pred)
            ref_tokens = self.vietnamese_metrics.vietnamese_tokenize(ref)
            
            try:
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return {'meteor': np.mean(meteor_scores)}
    
    def compute_coverage_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute content coverage metrics"""
        coverage_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.vietnamese_metrics.vietnamese_tokenize(pred))
            ref_tokens = set(self.vietnamese_metrics.vietnamese_tokenize(ref))
            
            if len(ref_tokens) == 0:
                coverage_scores.append(0.0)
                continue
            
            # Coverage: how much of reference content is covered
            covered = len(pred_tokens.intersection(ref_tokens))
            coverage = covered / len(ref_tokens)
            coverage_scores.append(coverage)
        
        return {
            'content_coverage': np.mean(coverage_scores),
            'high_coverage_ratio': sum(1 for score in coverage_scores if score > 0.7) / len(coverage_scores)
        }
    
    def compute_novelty_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute novelty metrics (how much new content is generated)"""
        novelty_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.vietnamese_metrics.vietnamese_tokenize(pred))
            ref_tokens = set(self.vietnamese_metrics.vietnamese_tokenize(ref))
            
            if len(pred_tokens) == 0:
                novelty_scores.append(0.0)
                continue
            
            # Novelty: how much new content is generated
            novel = len(pred_tokens - ref_tokens)
            novelty = novel / len(pred_tokens)
            novelty_scores.append(novelty)
        
        return {
            'novelty_score': np.mean(novelty_scores),
            'balanced_novelty_ratio': sum(1 for score in novelty_scores if 0.1 < score < 0.5) / len(novelty_scores)
        }
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        if len(predictions) != len(references):
            raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
        
        all_metrics = {}
        
        # Core metrics
        if 'rouge' in self.metrics:
            rouge_scores = self.compute_rouge_scores(predictions, references)
            all_metrics.update(rouge_scores)
        
        if 'bleu' in self.metrics:
            bleu_scores = self.compute_bleu_scores(predictions, references)
            all_metrics.update(bleu_scores)
        
        if 'bertscore' in self.metrics:
            bertscore_scores = self.compute_bertscore(predictions, references)
            all_metrics.update(bertscore_scores)
        
        if 'meteor' in self.metrics:
            meteor_scores = self.compute_meteor_scores(predictions, references)
            all_metrics.update(meteor_scores)
        
        # Additional metrics
        length_metrics = self.vietnamese_metrics.compute_length_metrics(predictions, references)
        all_metrics.update(length_metrics)
        
        repetition_metrics = self.vietnamese_metrics.compute_repetition_metrics(predictions)
        all_metrics.update(repetition_metrics)
        
        coverage_metrics = self.compute_coverage_metrics(predictions, references)
        all_metrics.update(coverage_metrics)
        
        novelty_metrics = self.compute_novelty_metrics(predictions, references)
        all_metrics.update(novelty_metrics)
        
        # Composite scores
        all_metrics['composite_score'] = self._compute_composite_score(all_metrics)
        
        return all_metrics
    
    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute a composite score combining multiple metrics"""
        # Weights for different metrics
        weights = {
            'rouge1_fmeasure': 0.25,
            'rouge2_fmeasure': 0.25,
            'rougeL_fmeasure': 0.25,
            'bleu_4': 0.15,
            'bertscore_f1': 0.1
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                composite += metrics[metric] * weight
                total_weight += weight
        
        return composite / max(total_weight, 1.0)
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """Print a formatted summary of metrics"""
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        # Core metrics
        print("\n📊 Core Metrics:")
        core_metrics = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure', 'bleu_4', 'bertscore_f1']
        for metric in core_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.4f}")
        
        # Quality metrics
        print("\n🎯 Quality Metrics:")
        quality_metrics = ['content_coverage', 'repetition_score', 'novelty_score']
        for metric in quality_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.4f}")
        
        # Length metrics
        print("\n📏 Length Metrics:")
        length_metrics = ['avg_pred_length', 'avg_ref_length', 'length_ratio']
        for metric in length_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.2f}")
        
        # Composite score
        if 'composite_score' in metrics:
            print(f"\n🏆 Composite Score: {metrics['composite_score']:.4f}")
        
        print("="*60)


def evaluate_model_predictions(predictions_file: str, references_file: str, config: Dict) -> Dict[str, float]:
    """Evaluate model predictions from files"""
    # Load predictions and references
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    # Create evaluator and compute metrics
    evaluator = SummarizationEvaluator(config)
    metrics = evaluator.compute_metrics(predictions, references)
    
    # Print summary
    evaluator.print_metrics_summary(metrics)
    
    return metrics
