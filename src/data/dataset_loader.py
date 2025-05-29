"""
Vietnamese News Dataset Loader and Preprocessor
Handles Vietnews dataset and other Vietnamese news sources
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, load_dataset
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
import logging
from underthesea import word_tokenize, sent_tokenize
import requests
from bs4 import BeautifulSoup


class VietnameseTextPreprocessor:
    """Preprocessor for Vietnamese text with advanced cleaning and normalization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # Vietnamese-specific patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+84|0)[0-9]{9,10}')
        
        # Vietnamese diacritics normalization
        self.diacritic_map = self._create_diacritic_map()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_diacritic_map(self) -> Dict[str, str]:
        """Create mapping for Vietnamese diacritic normalization"""
        return {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text"""
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
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters (optional)
        if self.config.get('remove_special_chars', False):
            text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        return text
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using underthesea"""
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            self.logger.warning(f"Tokenization failed: {e}")
            return text.split()
    
    def preprocess_article(self, article: str, summary: str) -> Tuple[str, str]:
        """Preprocess article and summary pair"""
        # Clean both texts
        clean_article = self.clean_text(article)
        clean_summary = self.clean_text(summary)
        
        # Check length constraints
        min_input_len = self.config.get('min_input_length', 50)
        max_input_len = self.config.get('max_input_length', 1024)
        min_summary_len = self.config.get('min_summary_length', 10)
        max_summary_len = self.config.get('max_summary_length', 256)
        
        if (len(clean_article.split()) < min_input_len or 
            len(clean_article.split()) > max_input_len or
            len(clean_summary.split()) < min_summary_len or
            len(clean_summary.split()) > max_summary_len):
            return None, None
        
        return clean_article, clean_summary


class VietnewsDataset(Dataset):
    """Dataset class for Vietnamese news summarization"""
    
    def __init__(self, data: List[Dict], tokenizer, config: Dict, mode: str = 'train'):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.preprocessor = VietnameseTextPreprocessor(config.get('preprocessing', {}))
        
        # Filter and preprocess data
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self) -> List[Dict]:
        """Preprocess all data samples"""
        processed = []
        
        for item in self.data:
            article = item.get('article', item.get('text', ''))
            summary = item.get('summary', item.get('target', ''))
            
            clean_article, clean_summary = self.preprocessor.preprocess_article(article, summary)
            
            if clean_article and clean_summary:
                processed.append({
                    'article': clean_article,
                    'summary': clean_summary,
                    'length': len(clean_article.split())
                })
        
        # Sort by length for curriculum learning
        if self.mode == 'train' and self.config.get('curriculum_learning', {}).get('enabled', False):
            processed.sort(key=lambda x: x['length'])
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            item['article'],
            max_length=self.config.get('max_input_length', 1024),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            item['summary'],
            max_length=self.config.get('max_output_length', 256),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'decoder_attention_mask': targets['attention_mask'].squeeze(),
            'length': item['length']
        }


class VietnameseNewsDataLoader:
    """Data loader for Vietnamese news datasets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['base_model']
        )
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_vietnews_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and split Vietnews dataset"""
        try:
            # Try to load from Hugging Face datasets
            dataset = load_dataset("vietnews", split="train")
            data = [{"article": item["article"], "summary": item["summary"]} for item in dataset]
        except Exception as e:
            self.logger.warning(f"Could not load from HF datasets: {e}")
            # Fallback to manual loading
            data = self._load_manual_vietnews()
        
        # Split data
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        total_size = len(data)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        self.logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _load_manual_vietnews(self) -> List[Dict]:
        """Manual loading of Vietnamese news data"""
        # This is a placeholder - you would implement actual data loading here
        # For now, return sample data
        sample_data = [
            {
                "article": "Hôm nay, Thủ tướng Chính phủ đã có cuộc họp quan trọng về tình hình kinh tế...",
                "summary": "Thủ tướng họp về kinh tế"
            }
        ] * 100  # Sample data for testing
        
        return sample_data
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders"""
        train_data, val_data, test_data = self.load_vietnews_dataset()
        
        # Create datasets
        train_dataset = VietnewsDataset(train_data, self.tokenizer, self.config, mode='train')
        val_dataset = VietnewsDataset(val_data, self.tokenizer, self.config, mode='val')
        test_dataset = VietnewsDataset(test_data, self.tokenizer, self.config, mode='test')
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config.get('hardware', {}).get('dataloader_num_workers', 4),
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('hardware', {}).get('dataloader_num_workers', 4),
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('hardware', {}).get('dataloader_num_workers', 4),
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True)
        )
        
        return train_loader, val_loader, test_loader


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function to create data loaders"""
    data_loader = VietnameseNewsDataLoader(config)
    return data_loader.create_dataloaders()
