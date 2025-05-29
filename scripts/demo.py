#!/usr/bin/env python3
"""
Quick Demo Script for Vietnamese Summarization

Usage:
    python scripts/demo.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.enhanced_t5 import create_enhanced_model
from src.data.dataset import load_sample_data
from src.evaluation.metrics import SummarizationEvaluator
from src.utils.config import get_default_config


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def generate_summary(model, text: str) -> str:
    """Generate summary for given text."""
    try:
        inputs = model.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4,
            length_penalty=2.0,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        summary = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        return f"Error: {e}"


def main():
    """Main demo function."""
    logger = setup_logging()
    
    logger.info("🚀 Vietnamese Summarization Demo")
    logger.info("="*50)
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Load sample data
        logger.info("📚 Loading sample data...")
        sample_data = load_sample_data()
        logger.info(f"Loaded {len(sample_data)} sample articles")
        
        # Create model (this might take a while)
        logger.info("🤖 Loading model (this may take a few minutes)...")
        model = create_enhanced_model(config['model']['base_model'])
        logger.info("✅ Model loaded successfully!")
        
        # Initialize evaluator
        evaluator = SummarizationEvaluator(config)
        
        # Process samples
        logger.info("\n" + "="*50)
        logger.info("🔍 PROCESSING SAMPLE ARTICLES")
        logger.info("="*50)
        
        predictions = []
        references = []
        
        for i, sample in enumerate(sample_data[:3]):  # Process first 3 samples
            logger.info(f"\n📰 Sample {i+1}:")
            logger.info("-" * 30)
            
            article = sample['article']
            reference = sample['summary']
            
            # Show original
            logger.info(f"📄 Article ({len(article.split())} words):")
            logger.info(article[:200] + "..." if len(article) > 200 else article)
            
            logger.info(f"\n📝 Reference ({len(reference.split())} words):")
            logger.info(reference)
            
            # Generate summary
            logger.info("\n🤖 Generating summary...")
            generated = generate_summary(model, article)
            
            logger.info(f"✨ Generated ({len(generated.split())} words):")
            logger.info(generated)
            
            predictions.append(generated)
            references.append(reference)
            
            logger.info("-" * 30)
        
        # Evaluate results
        if predictions and references:
            logger.info("\n" + "="*50)
            logger.info("📊 EVALUATION RESULTS")
            logger.info("="*50)
            
            metrics = evaluator.compute_metrics(predictions, references)
            evaluator.print_metrics_summary(metrics)
        
        logger.info("\n✅ Demo completed successfully!")
        logger.info("💡 To train your own model: python scripts/train.py")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.info("💡 Make sure you have internet connection for model download")


if __name__ == "__main__":
    main()
