"""
Vietnamese Dataset for Summarization

Handles Vietnamese text preprocessing, tokenization, and data loading
with support for curriculum learning and quality filtering.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import Vietnamese NLP tools
try:
    from underthesea import word_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logger.warning("underthesea not available. Using basic tokenization.")


class VietnameseTextPreprocessor:
    """Preprocessor for Vietnamese text with cleaning and normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://[^\s]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+84|0)[0-9]{9,10}')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Normalize unicode
        if self.config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenize Vietnamese text."""
        if not text:
            return []
        
        if UNDERTHESEA_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback to basic tokenization
        return text.split()
    
    def preprocess_article(self, article: str, summary: str) -> Tuple[Optional[str], Optional[str]]:
        """Preprocess article and summary with quality filtering."""
        clean_article = self.clean_text(article)
        clean_summary = self.clean_text(summary)
        
        if not clean_article or not clean_summary:
            return None, None
        
        # Check length constraints
        min_input_len = self.config.get('min_input_length', 50)
        max_input_len = self.config.get('max_input_length', 1024)
        min_summary_len = self.config.get('min_summary_length', 10)
        max_summary_len = self.config.get('max_summary_length', 256)
        
        article_words = len(clean_article.split())
        summary_words = len(clean_summary.split())
        
        if (article_words < min_input_len or article_words > max_input_len or
            summary_words < min_summary_len or summary_words > max_summary_len):
            return None, None
        
        return clean_article, clean_summary


class VietnameseDataset(Dataset):
    """PyTorch Dataset for Vietnamese news summarization."""
    
    def __init__(self, data: List[Dict], tokenizer: Any, config: Dict, mode: str = 'train'):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
        # Initialize preprocessor
        preprocessing_config = config.get('data', {}).get('preprocessing', {})
        self.preprocessor = VietnameseTextPreprocessor(preprocessing_config)
        
        # Process data
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Initialized {mode} dataset with {len(self.processed_data)} samples")
    
    def _preprocess_data(self) -> List[Dict]:
        """Preprocess and filter data."""
        processed = []
        
        for item in self.data:
            article = item.get('article', item.get('text', ''))
            summary = item.get('summary', item.get('target', ''))
            
            if not article or not summary:
                continue
            
            clean_article, clean_summary = self.preprocessor.preprocess_article(article, summary)
            
            if clean_article and clean_summary:
                processed.append({
                    'article': clean_article,
                    'summary': clean_summary,
                    'length': len(clean_article.split())
                })
        
        # Sort by length for curriculum learning
        if self.mode == 'train' and self.config.get('training', {}).get('curriculum_learning', False):
            processed.sort(key=lambda x: x['length'])
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Get max lengths from config
        max_input_length = self.config.get('model', {}).get('max_input_length', 512)
        max_output_length = self.config.get('model', {}).get('max_output_length', 128)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            item['article'],
            max_length=max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            item['summary'],
            max_length=max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'decoder_attention_mask': targets['attention_mask'].squeeze()
        }


def load_sample_data() -> List[Dict[str, str]]:
    """Load sample Vietnamese news data for testing."""
    return [
        {
            "article": """Hôm nay (15/11), Thủ tướng Chính phủ Phạm Minh Chính đã chủ trì cuộc họp Chính phủ thường kỳ tháng 11/2024. Tại cuộc họp, Thủ tướng nhấn mạnh tình hình kinh tế vĩ mô tiếp tục ổn định, lạm phát được kiểm soát tốt. Tăng trưởng GDP 9 tháng đầu năm đạt 6.82%, cao hơn mục tiêu đề ra. Xuất khẩu tiếp tục tăng trưởng tích cực với kim ngạch đạt 334.5 tỷ USD, tăng 14.9% so với cùng kỳ năm trước. Thủ tướng cũng chỉ đạo các bộ, ngành tập trung thực hiện các giải pháp thúc đẩy tăng trưởng trong những tháng cuối năm.""",
            "summary": "Thủ tướng chủ trì họp Chính phủ, nhấn mạnh kinh tế vĩ mô ổn định, GDP 9 tháng đạt 6.82%, xuất khẩu tăng 14.9%."
        },
        {
            "article": """Ngân hàng Nhà nước Việt Nam (NHNN) vừa công bố báo cáo tình hình hoạt động ngân hàng quý III/2024. Theo đó, tổng tài sản của hệ thống các tổ chức tín dụng đạt 15.2 triệu tỷ đồng, tăng 8.7% so với đầu năm. Dư nợ tín dụng toàn nền kinh tế tăng 8.9% so với cuối năm 2023, đạt 13.8 triệu tỷ đồng. NHNN cho biết chất lượng tín dụng tiếp tục được cải thiện, tỷ lệ nợ xấu của toàn hệ thống giảm xuống 4.55%.""",
            "summary": "NHNN báo cáo tài sản hệ thống tín dụng đạt 15.2 triệu tỷ đồng, dư nợ tăng 8.9%, nợ xấu giảm xuống 4.55%."
        },
        {
            "article": """Tập đoàn FPT vừa công bố kết quả kinh doanh quý III/2024 với doanh thu đạt 9.6 nghìn tỷ đồng, tăng 20.1% so với cùng kỳ năm trước. Lợi nhuận sau thuế đạt 1.1 nghìn tỷ đồng, tăng 22.8%. Trong đó, mảng công nghệ đóng góp 7.8 nghìn tỷ đồng doanh thu, tăng 20.5%. FPT Software tiếp tục là động lực tăng trưởng chính với doanh thu đạt 4.2 nghìn tỷ đồng.""",
            "summary": "FPT báo cáo doanh thu quý III đạt 9.6 nghìn tỷ đồng, tăng 20.1%, lợi nhuận tăng 22.8%."
        }
    ] * 10  # Replicate for testing


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Load sample data
    sample_data = load_sample_data()
    
    # Split data
    total_size = len(sample_data)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    # Load tokenizer
    model_name = config.get('model', {}).get('base_model', 'VietAI/vit5-large-vietnews-summarization')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = VietnameseDataset(train_data, tokenizer, config, mode='train')
    val_dataset = VietnameseDataset(val_data, tokenizer, config, mode='val')
    test_dataset = VietnameseDataset(test_data, tokenizer, config, mode='test')
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
