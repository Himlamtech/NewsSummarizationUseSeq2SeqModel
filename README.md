# Vietnamese News Summarization Enhancement
## Advanced NLP Techniques for Graduation Thesis

This project enhances the VietAI/vit5-large-vietnews-summarization model with advanced NLP techniques for Vietnamese text summarization.

## 🎯 Project Overview

### Core Enhancements
1. **Enhanced Attention Mechanisms**: Self-attention improvements for better context understanding
2. **Pointer-Generator Networks**: Handle rare words and Vietnamese proper nouns effectively
3. **Coverage Mechanism**: Reduce repetition and improve content coverage
4. **Curriculum Learning**: Progressive training from simple to complex texts

### Technical Stack
- **Base Model**: VietAI/vit5-large-vietnews-summarization
- **Framework**: PyTorch, Transformers
- **Dataset**: Vietnews (tuoitre.vn, vnexpress.net, nguoiduatin.vn)
- **Interface**: Streamlit
- **Evaluation**: ROUGE, BLEU, BERTScore

## 📁 Project Structure

```
PTITProject/NewsSumarize/
├── src/
│   ├── models/              # Enhanced model architectures
│   ├── data/               # Data processing and loading
│   ├── training/           # Training scripts and utilities
│   ├── evaluation/         # Evaluation metrics and analysis
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks for experiments
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── checkpoints/           # Model checkpoints
├── results/               # Evaluation results
├── streamlit_app/         # Web interface
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
```bash
python src/data/prepare_dataset.py
```

### Training
```bash
python src/training/train_enhanced_model.py --config configs/enhanced_config.yaml
```

### Evaluation
```bash
python src/evaluation/evaluate_model.py --model_path checkpoints/best_model.pt
```

### Demo
```bash
streamlit run streamlit_app/app.py
```

## 📊 Key Features

### 1. Enhanced Attention Mechanisms
- **Self-Attention Improvements**: Better long-range dependency modeling
- **Attention Visualization**: Interactive attention maps in the demo

### 2. Pointer-Generator Networks
- **Copy Mechanism**: Handle out-of-vocabulary words
- **Vietnamese Proper Nouns**: Preserve important named entities

### 3. Coverage Mechanism
- **Repetition Reduction**: Track covered content during generation
- **Content Coverage**: Ensure comprehensive summarization

### 4. Advanced Training
- **Transfer Learning**: Build upon VietAI's pre-trained model
- **Curriculum Learning**: Progressive difficulty training
- **Teacher Forcing**: Scheduled sampling for better generalization

## 📈 Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore |
|-------|---------|---------|---------|------|-----------|
| Baseline (VietAI) | - | - | - | - | - |
| Enhanced Model | - | - | - | - | - |

## 🔬 Ablation Studies

Detailed analysis of each enhancement's contribution to model performance.

## 📝 Citation

```bibtex
@thesis{vietnamese_summarization_enhancement,
  title={Enhanced Vietnamese News Summarization with Advanced NLP Techniques},
  author={Your Name},
  year={2025},
  school={PTIT}
}
```

## 📄 License

MIT License
