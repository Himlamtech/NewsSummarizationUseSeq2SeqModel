"""
Enhanced Vietnamese Text Summarization

A graduation thesis project implementing advanced NLP techniques for Vietnamese news summarization.
"""

__version__ = "1.0.0"
__author__ = "PTIT Graduation Thesis"

from .models.enhanced_t5 import EnhancedT5Model, create_enhanced_model
from .data.dataset import VietnameseDataset, create_data_loaders
from .evaluation.metrics import SummarizationEvaluator
from .training.trainer import EnhancedTrainer

__all__ = [
    "EnhancedT5Model",
    "create_enhanced_model",
    "VietnameseDataset", 
    "create_data_loaders",
    "SummarizationEvaluator",
    "EnhancedTrainer"
]
