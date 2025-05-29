"""
Data Augmentation for Vietnamese Summarization using GPT-4
Implements various augmentation strategies for improving model robustness
"""

import openai
import asyncio
import aiohttp
import json
import random
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import time
import os
from underthesea import word_tokenize, sent_tokenize


class VietnameseDataAugmenter:
    """Data augmentation for Vietnamese text using various strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # OpenAI API setup (if available)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Augmentation strategies
        self.strategies = {
            'paraphrase': self.paraphrase_text,
            'sentence_reorder': self.reorder_sentences,
            'synonym_replacement': self.replace_synonyms,
            'back_translation': self.back_translate,
            'gpt4_augmentation': self.gpt4_augment
        }
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def paraphrase_text(self, text: str) -> str:
        """Paraphrase text using rule-based methods"""
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Simple paraphrasing rules for Vietnamese
            paraphrased = sentence
            
            # Replace some common phrases
            replacements = {
                'theo như': 'như',
                'bởi vì': 'vì',
                'do đó': 'vậy nên',
                'tuy nhiên': 'nhưng',
                'ngoài ra': 'bên cạnh đó'
            }
            
            for old, new in replacements.items():
                if random.random() < 0.3:  # 30% chance to replace
                    paraphrased = paraphrased.replace(old, new)
            
            paraphrased_sentences.append(paraphrased)
        
        return ' '.join(paraphrased_sentences)
    
    def reorder_sentences(self, text: str) -> str:
        """Reorder sentences while maintaining coherence"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return text
        
        # Keep first and last sentences in place, shuffle middle ones
        if len(sentences) > 3:
            middle_sentences = sentences[1:-1]
            random.shuffle(middle_sentences)
            reordered = [sentences[0]] + middle_sentences + [sentences[-1]]
        else:
            reordered = sentences
        
        return ' '.join(reordered)
    
    def replace_synonyms(self, text: str) -> str:
        """Replace words with Vietnamese synonyms"""
        # Vietnamese synonym dictionary (simplified)
        synonyms = {
            'lớn': ['to', 'khổng lồ', 'rộng lớn'],
            'nhỏ': ['bé', 'tí hon', 'nhỏ bé'],
            'tốt': ['hay', 'giỏi', 'xuất sắc'],
            'xấu': ['dở', 'tệ', 'không tốt'],
            'nhanh': ['mau', 'nhanh chóng', 'thần tốc'],
            'chậm': ['từ từ', 'chậm chạp', 'ì ạch'],
            'quan trọng': ['thiết yếu', 'cần thiết', 'then chốt'],
            'khó khăn': ['vất vả', 'gian nan', 'khó nhọc']
        }
        
        words = word_tokenize(text)
        augmented_words = []
        
        for word in words:
            if word.lower() in synonyms and random.random() < 0.2:  # 20% chance
                synonym = random.choice(synonyms[word.lower()])
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def back_translate(self, text: str) -> str:
        """Back translation through English (simplified simulation)"""
        # This is a simplified version - in practice, you'd use translation APIs
        # For now, we'll just apply some transformations that simulate back-translation effects
        
        # Common back-translation artifacts in Vietnamese
        transformations = [
            ('của', 'thuộc về'),
            ('và', 'cùng với'),
            ('nhưng', 'tuy nhiên'),
            ('rất', 'vô cùng'),
            ('có thể', 'có khả năng')
        ]
        
        result = text
        for old, new in transformations:
            if random.random() < 0.3:
                result = result.replace(old, new)
        
        return result
    
    async def gpt4_augment(self, text: str) -> str:
        """Augment text using GPT-4 API"""
        if not self.openai_api_key:
            self.logger.warning("OpenAI API key not found, skipping GPT-4 augmentation")
            return text
        
        try:
            prompt = f"""
            Hãy viết lại đoạn văn sau bằng tiếng Việt, giữ nguyên ý nghĩa nhưng thay đổi cách diễn đạt:
            
            "{text}"
            
            Yêu cầu:
            - Giữ nguyên thông tin chính
            - Thay đổi cấu trúc câu và từ ngữ
            - Văn phong tự nhiên, phù hợp với báo chí Việt Nam
            - Độ dài tương tự văn bản gốc
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia viết lại văn bản tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            augmented_text = response.choices[0].message.content.strip()
            return augmented_text
            
        except Exception as e:
            self.logger.error(f"GPT-4 augmentation failed: {e}")
            return text
    
    def augment_dataset(self, dataset: List[Dict], strategies: List[str], 
                       augmentation_ratio: float = 0.5) -> List[Dict]:
        """Augment entire dataset using specified strategies"""
        augmented_data = []
        original_size = len(dataset)
        target_size = int(original_size * (1 + augmentation_ratio))
        
        self.logger.info(f"Augmenting dataset from {original_size} to {target_size} samples")
        
        # Add original data
        augmented_data.extend(dataset)
        
        # Generate augmented samples
        samples_to_generate = target_size - original_size
        
        for i in range(samples_to_generate):
            # Select random sample and strategy
            original_sample = random.choice(dataset)
            strategy = random.choice(strategies)
            
            try:
                # Augment article
                if strategy in self.strategies:
                    augmented_article = self.strategies[strategy](original_sample['article'])
                    
                    # For summary, use simpler augmentation to maintain quality
                    if strategy == 'gpt4_augmentation':
                        augmented_summary = original_sample['summary']  # Keep original summary
                    else:
                        augmented_summary = self.paraphrase_text(original_sample['summary'])
                    
                    augmented_sample = {
                        'article': augmented_article,
                        'summary': augmented_summary,
                        'augmentation_strategy': strategy,
                        'original_id': original_sample.get('id', i)
                    }
                    
                    augmented_data.append(augmented_sample)
                    
            except Exception as e:
                self.logger.error(f"Augmentation failed for sample {i}: {e}")
                continue
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{samples_to_generate} augmented samples")
        
        self.logger.info(f"Dataset augmentation completed. Final size: {len(augmented_data)}")
        return augmented_data
    
    def quality_filter(self, original: str, augmented: str, 
                      min_similarity: float = 0.3, max_similarity: float = 0.9) -> bool:
        """Filter augmented samples based on quality metrics"""
        # Simple similarity check using word overlap
        original_words = set(word_tokenize(original.lower()))
        augmented_words = set(word_tokenize(augmented.lower()))
        
        if len(original_words) == 0 or len(augmented_words) == 0:
            return False
        
        # Jaccard similarity
        intersection = len(original_words.intersection(augmented_words))
        union = len(original_words.union(augmented_words))
        similarity = intersection / union if union > 0 else 0
        
        # Length ratio check
        length_ratio = len(augmented_words) / len(original_words)
        
        return (min_similarity <= similarity <= max_similarity and 
                0.5 <= length_ratio <= 2.0)
    
    def save_augmented_dataset(self, augmented_data: List[Dict], output_path: str):
        """Save augmented dataset to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Augmented dataset saved to {output_path}")
    
    def load_augmented_dataset(self, input_path: str) -> List[Dict]:
        """Load augmented dataset from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} samples from {input_path}")
        return data


async def main():
    """Example usage of data augmentation"""
    config = {
        'augmentation': {
            'strategies': ['paraphrase', 'sentence_reorder', 'synonym_replacement'],
            'ratio': 0.5,
            'quality_filter': True
        }
    }
    
    # Sample data
    sample_data = [
        {
            'article': 'Hôm nay, chính phủ đã công bố kế hoạch phát triển kinh tế mới...',
            'summary': 'Chính phủ công bố kế hoạch kinh tế mới',
            'id': 1
        }
    ]
    
    # Create augmenter
    augmenter = VietnameseDataAugmenter(config)
    
    # Augment dataset
    augmented_data = augmenter.augment_dataset(
        sample_data, 
        strategies=['paraphrase', 'sentence_reorder'],
        augmentation_ratio=0.5
    )
    
    # Save results
    augmenter.save_augmented_dataset(augmented_data, 'data/augmented_dataset.json')


if __name__ == "__main__":
    asyncio.run(main())
