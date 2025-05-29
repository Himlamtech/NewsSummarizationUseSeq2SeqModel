#!/usr/bin/env python3
"""
Test Script for Vietnamese Summarization Installation

Usage:
    python scripts/test.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing imports...")
    
    try:
        # Test core imports
        import torch
        logger.info("  ✅ PyTorch")
        
        from transformers import AutoTokenizer
        logger.info("  ✅ Transformers")
        
        # Test package imports
        from src.models.enhanced_t5 import create_enhanced_model
        logger.info("  ✅ Enhanced T5 Model")
        
        from src.data.dataset import load_sample_data, create_data_loaders
        logger.info("  ✅ Data Loading")
        
        from src.evaluation.metrics import SummarizationEvaluator
        logger.info("  ✅ Evaluation Metrics")
        
        from src.training.trainer import EnhancedTrainer
        logger.info("  ✅ Training")
        
        from src.utils.config import get_default_config
        logger.info("  ✅ Configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Import failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing data loading...")
    
    try:
        from src.data.dataset import load_sample_data
        from src.utils.config import get_default_config
        
        # Load sample data
        sample_data = load_sample_data()
        logger.info(f"  ✅ Loaded {len(sample_data)} sample articles")
        
        # Test data format
        if sample_data and 'article' in sample_data[0] and 'summary' in sample_data[0]:
            logger.info("  ✅ Data format is correct")
        else:
            logger.error("  ❌ Invalid data format")
            return False
        
        # Test data loaders creation
        config = get_default_config()
        from src.data.dataset import create_data_loaders
        
        train_loader, val_loader, test_loader = create_data_loaders(config)
        logger.info(f"  ✅ Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Data loading failed: {e}")
        return False


def test_evaluation():
    """Test evaluation functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing evaluation...")
    
    try:
        from src.evaluation.metrics import SummarizationEvaluator
        from src.utils.config import get_default_config
        
        config = get_default_config()
        evaluator = SummarizationEvaluator(config)
        
        # Test with sample data
        predictions = ["Đây là một bản tóm tắt mẫu."]
        references = ["Đây là bản tóm tắt tham khảo."]
        
        metrics = evaluator.compute_metrics(predictions, references)
        logger.info(f"  ✅ Computed {len(metrics)} metrics")
        
        # Check key metrics exist
        if 'composite_score' in metrics:
            logger.info(f"  ✅ Composite score: {metrics['composite_score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Evaluation failed: {e}")
        return False


def test_configuration():
    """Test configuration functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing configuration...")
    
    try:
        from src.utils.config import get_default_config, load_config
        
        # Test default config
        config = get_default_config()
        logger.info("  ✅ Default configuration loaded")
        
        # Check required sections
        required_sections = ['model', 'data', 'training', 'evaluation']
        for section in required_sections:
            if section in config:
                logger.info(f"  ✅ {section} section exists")
            else:
                logger.error(f"  ❌ {section} section missing")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Configuration failed: {e}")
        return False


def test_directories():
    """Test that required directories can be created."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing directories...")
    
    try:
        required_dirs = ['checkpoints', 'results', 'logs']
        
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name):
                logger.info(f"  ✅ {dir_name}/ created")
            else:
                logger.error(f"  ❌ Failed to create {dir_name}/")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Directory creation failed: {e}")
        return False


def main():
    """Run all tests."""
    logger = setup_logging()
    
    logger.info("🚀 Vietnamese Summarization Installation Test")
    logger.info("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Directories", test_directories),
        ("Data Loading", test_data_loading),
        ("Evaluation", test_evaluation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:15}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Installation is working correctly.")
        logger.info("\n💡 Next steps:")
        logger.info("  1. Run demo: python scripts/demo.py")
        logger.info("  2. Start training: python scripts/train.py")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
