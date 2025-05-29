"""
Enhanced Models for Vietnamese Summarization

This module contains the enhanced T5 model with advanced attention mechanisms,
pointer-generator networks, and coverage mechanisms specifically designed
for Vietnamese text summarization.
"""

from .enhanced_t5 import (
    EnhancedSelfAttention,
    PointerGeneratorNetwork, 
    CoverageMechanism,
    EnhancedT5ForConditionalGeneration,
    create_enhanced_model
)

__all__ = [
    "EnhancedSelfAttention",
    "PointerGeneratorNetwork",
    "CoverageMechanism", 
    "EnhancedT5ForConditionalGeneration",
    "create_enhanced_model"
]
