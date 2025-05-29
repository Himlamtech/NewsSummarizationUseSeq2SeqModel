#!/usr/bin/env python3
"""
Setup Script for Enhanced Vietnamese Summarization Project
Handles installation, data preparation, and initial setup
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import requests
import zipfile
import json
from typing import List, Dict


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(command: List[str], logger, check=True):
    """Run shell command with logging"""
    logger.info(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(e.stderr)
        raise


def install_dependencies(logger):
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    
    # Upgrade pip first
    run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], logger)
    
    # Install requirements
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        run_command([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], logger)
    else:
        logger.warning(f"Requirements file {requirements_file} not found")
        
        # Install essential packages manually
        essential_packages = [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'datasets>=2.12.0',
            'streamlit>=1.24.0',
            'underthesea>=6.0.0',
            'rouge-score>=0.1.2',
            'bert-score>=0.3.13',
            'wandb>=0.15.0',
            'pyyaml>=6.0',
            'tqdm>=4.65.0',
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'matplotlib>=3.7.0',
            'plotly>=5.14.0'
        ]
        
        for package in essential_packages:
            try:
                run_command([sys.executable, '-m', 'pip', 'install', package], logger)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {package}, continuing...")


def setup_directories(logger):
    """Create necessary directories"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'checkpoints',
        'results',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_sample_data(logger):
    """Download sample Vietnamese news data"""
    logger.info("Setting up sample data...")
    
    # Create sample data for testing
    sample_data = [
        {
            "id": 1,
            "article": """Hôm nay (15/11), Thủ tướng Chính phủ Phạm Minh Chính đã chủ trì cuộc họp Chính phủ thường kỳ tháng 11/2024. Tại cuộc họp, Thủ tướng nhấn mạnh tình hình kinh tế vĩ mô tiếp tục ổn định, lạm phát được kiểm soát tốt. Tăng trưởng GDP 9 tháng đầu năm đạt 6.82%, cao hơn mục tiêu đề ra. Xuất khẩu tiếp tục tăng trưởng tích cực với kim ngạch đạt 334.5 tỷ USD, tăng 14.9% so với cùng kỳ năm trước. Thủ tướng cũng chỉ đạo các bộ, ngành tập trung thực hiện các giải pháp thúc đẩy tăng trưởng trong những tháng cuối năm, đặc biệt là đẩy mạnh giải ngân đầu tư công và hỗ trợ doanh nghiệp.""",
            "summary": "Thủ tướng chủ trì họp Chính phủ, nhấn mạnh kinh tế vĩ mô ổn định, GDP 9 tháng đạt 6.82%, xuất khẩu tăng 14.9%."
        },
        {
            "id": 2,
            "article": """Ngân hàng Nhà nước Việt Nam (NHNN) vừa công bố báo cáo tình hình hoạt động ngân hàng quý III/2024. Theo đó, tổng tài sản của hệ thống các tổ chức tín dụng đạt 15.2 triệu tỷ đồng, tăng 8.7% so với đầu năm. Dư nợ tín dụng toàn nền kinh tế tăng 8.9% so với cuối năm 2023, đạt 13.8 triệu tỷ đồng. NHNN cho biết chất lượng tín dụng tiếp tục được cải thiện, tỷ lệ nợ xấu của toàn hệ thống giảm xuống 4.55%. Hoạt động ngân hàng diễn ra ổn định, thanh khoản được đảm bảo, lãi suất huy động và cho vay có xu hướng giảm nhẹ, tạo điều kiện thuận lợi cho doanh nghiệp và người dân tiếp cận vốn.""",
            "summary": "NHNN báo cáo tài sản hệ thống tín dụng đạt 15.2 triệu tỷ đồng, dư nợ tăng 8.9%, nợ xấu giảm xuống 4.55%."
        },
        {
            "id": 3,
            "article": """Tập đoàn FPT vừa công bố kết quả kinh doanh quý III/2024 với doanh thu đạt 9.6 nghìn tỷ đồng, tăng 20.1% so với cùng kỳ năm trước. Lợi nhuận sau thuế đạt 1.1 nghìn tỷ đồng, tăng 22.8%. Trong đó, mảng công nghệ đóng góp 7.8 nghìn tỷ đồng doanh thu, tăng 20.5%. FPT Software tiếp tục là động lực tăng trưởng chính với doanh thu đạt 4.2 nghìn tỷ đồng. Công ty cũng thông báo kế hoạch mở rộng hoạt động tại thị trường Mỹ và châu Âu, dự kiến sẽ tuyển dụng thêm 10,000 nhân viên trong năm 2025. FPT đặt mục tiêu doanh thu năm 2024 đạt 38 nghìn tỷ đồng.""",
            "summary": "FPT báo cáo doanh thu quý III đạt 9.6 nghìn tỷ đồng, tăng 20.1%, lợi nhuận tăng 22.8%, kế hoạch tuyển 10,000 nhân viên."
        }
    ]
    
    # Save sample data
    data_dir = Path('data/raw')
    sample_file = data_dir / 'sample_vietnews.json'
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Sample data saved to {sample_file}")
    
    # Create train/val/test splits
    train_data = sample_data[:2]
    val_data = sample_data[2:3]
    test_data = sample_data[2:3]  # Same as val for demo
    
    processed_dir = Path('data/processed')
    
    for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_file = processed_dir / f'{split}.json'
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Created {split} split with {len(data)} samples")


def setup_wandb(logger):
    """Setup Weights & Biases for experiment tracking"""
    try:
        import wandb
        logger.info("Weights & Biases is available for experiment tracking")
        logger.info("To use wandb, run: wandb login")
    except ImportError:
        logger.warning("Weights & Biases not installed. Install with: pip install wandb")


def setup_git_hooks(logger):
    """Setup git hooks for code quality"""
    if not os.path.exists('.git'):
        logger.info("Not a git repository, skipping git hooks setup")
        return
    
    hooks_dir = Path('.git/hooks')
    
    # Pre-commit hook for code formatting
    pre_commit_hook = hooks_dir / 'pre-commit'
    
    hook_content = """#!/bin/bash
# Auto-format Python code before commit
echo "Running code formatting..."

# Format with black
black --check --diff src/ || {
    echo "Code formatting issues found. Run 'black src/' to fix."
    exit 1
}

# Check with flake8
flake8 src/ || {
    echo "Code style issues found. Please fix before committing."
    exit 1
}

echo "Code quality checks passed!"
"""
    
    try:
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        logger.info("Git pre-commit hook installed")
    except Exception as e:
        logger.warning(f"Failed to setup git hooks: {e}")


def verify_installation(logger):
    """Verify that installation was successful"""
    logger.info("Verifying installation...")
    
    # Check Python packages
    required_packages = [
        'torch',
        'transformers',
        'streamlit',
        'underthesea',
        'rouge_score',
        'bert_score'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    # Check directories
    required_dirs = ['data', 'checkpoints', 'results', 'logs']
    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✓ Directory: {directory}")
        else:
            logger.error(f"✗ Directory: {directory}")
            return False
    
    # Check configuration file
    if os.path.exists('configs/enhanced_config.yaml'):
        logger.info("✓ Configuration file")
    else:
        logger.error("✗ Configuration file")
        return False
    
    logger.info("Installation verification completed successfully!")
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Enhanced Vietnamese Summarization Project")
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data setup'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Setup development environment with additional tools'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting Enhanced Vietnamese Summarization Project Setup")
    
    try:
        # Setup directories
        setup_directories(logger)
        
        # Install dependencies
        if not args.skip_deps:
            install_dependencies(logger)
        
        # Setup sample data
        if not args.skip_data:
            download_sample_data(logger)
        
        # Setup development tools
        if args.dev:
            setup_wandb(logger)
            setup_git_hooks(logger)
        
        # Verify installation
        if verify_installation(logger):
            logger.info("Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run training: python train.py")
            logger.info("2. Start demo: streamlit run streamlit_app/app.py")
            logger.info("3. Check notebooks/ for examples")
        else:
            logger.error("Setup verification failed. Please check the errors above.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
