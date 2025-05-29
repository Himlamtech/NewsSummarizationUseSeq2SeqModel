"""
Vietnamese Summarization Package

Enhanced Vietnamese text summarization using advanced NLP techniques including:
- Self-attention improvements
- Pointer-generator networks  
- Coverage mechanisms
- Curriculum learning

This package provides a complete framework for Vietnamese news summarization
with state-of-the-art neural architectures and comprehensive evaluation.
"""

__version__ = "1.0.0"
__author__ = "PTIT Thesis Project"
__email__ = "your.email@ptit.edu.vn"

from .models import EnhancedT5ForConditionalGeneration, create_enhanced_model
from .data import VietnameseTextPreprocessor, VietnewsDataset, create_data_loaders
from .evaluation import SummarizationEvaluator
from .training import EnhancedTrainer
from .utils import VietnameseDataAugmenter

__all__ = [
    "EnhancedT5ForConditionalGeneration",
    "create_enhanced_model", 
    "VietnameseTextPreprocessor",
    "VietnewsDataset",
    "create_data_loaders",
    "SummarizationEvaluator",
    "EnhancedTrainer",
    "VietnameseDataAugmenter",
]
