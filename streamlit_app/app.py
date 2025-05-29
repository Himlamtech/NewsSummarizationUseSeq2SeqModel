"""
Streamlit Demo Application for Enhanced Vietnamese Summarization
Interactive interface with attention visualization and parameter tuning
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.enhanced_t5 import create_enhanced_model
from evaluation.metrics import SummarizationEvaluator
from data.dataset_loader import VietnameseTextPreprocessor

# Configure page
st.set_page_config(
    page_title="Vietnamese News Summarization",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .attention-heatmap {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class SummarizationDemo:
    """Main demo application class"""
    
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.initialize_components()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """Load configuration"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'enhanced_config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            st.error("Configuration file not found. Please ensure configs/enhanced_config.yaml exists.")
            self.config = {}
    
    @st.cache_resource
    def initialize_components(_self):
        """Initialize model and other components"""
        try:
            # Load model
            model_path = st.session_state.get('model_path', 'VietAI/vit5-large-vietnews-summarization')
            _self.model = create_enhanced_model(model_path)
            _self.model.eval()
            
            # Initialize preprocessor
            _self.preprocessor = VietnameseTextPreprocessor(_self.config.get('data', {}).get('preprocessing', {}))
            
            # Initialize evaluator
            _self.evaluator = SummarizationEvaluator(_self.config)
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return False
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🇻🇳 Enhanced Vietnamese News Summarization</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Model", "VietAI/vit5-large")
        with col2:
            st.metric("Enhancements", "3 Active")
        with col3:
            st.metric("Language", "Vietnamese")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🎛️ Generation Parameters")
        
        # Generation parameters
        generation_params = {}
        
        generation_params['num_beams'] = st.sidebar.slider(
            "Beam Size", min_value=1, max_value=10, value=4,
            help="Number of beams for beam search. Higher values may produce better results but are slower."
        )
        
        generation_params['length_penalty'] = st.sidebar.slider(
            "Length Penalty", min_value=0.5, max_value=3.0, value=2.0, step=0.1,
            help="Penalty for sequence length. Higher values encourage longer sequences."
        )
        
        generation_params['repetition_penalty'] = st.sidebar.slider(
            "Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1,
            help="Penalty for repeating tokens. Higher values reduce repetition."
        )
        
        generation_params['no_repeat_ngram_size'] = st.sidebar.slider(
            "No Repeat N-gram Size", min_value=0, max_value=5, value=3,
            help="Size of n-grams that cannot be repeated."
        )
        
        generation_params['max_length'] = st.sidebar.slider(
            "Max Summary Length", min_value=50, max_value=500, value=256,
            help="Maximum length of generated summary."
        )
        
        # Enhancement toggles
        st.sidebar.header("🔧 Model Enhancements")
        
        enhancements = {}
        enhancements['use_attention_viz'] = st.sidebar.checkbox(
            "Show Attention Visualization", value=True,
            help="Display attention heatmaps for model interpretability."
        )
        
        enhancements['use_coverage'] = st.sidebar.checkbox(
            "Coverage Mechanism", value=True,
            help="Use coverage mechanism to reduce repetition."
        )
        
        enhancements['use_pointer_gen'] = st.sidebar.checkbox(
            "Pointer-Generator", value=True,
            help="Use pointer-generator network for handling OOV words."
        )
        
        return generation_params, enhancements
    
    def generate_summary(self, text: str, generation_params: Dict, enhancements: Dict) -> Tuple[str, Dict]:
        """Generate summary with attention weights"""
        try:
            # Preprocess text
            clean_text, _ = self.preprocessor.preprocess_article(text, "dummy")
            if not clean_text:
                return "Error: Text preprocessing failed.", {}
            
            # Tokenize
            inputs = self.model.tokenizer(
                clean_text,
                max_length=self.config.get('model', {}).get('max_input_length', 1024),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Generate with attention
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict_in_generate=True,
                    **generation_params
                )
            
            # Decode summary
            summary = self.model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract attention weights
            attention_data = {}
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                attention_weights = outputs.cross_attentions[0][0].cpu().numpy()  # First layer, first head
                attention_data['weights'] = attention_weights
                attention_data['input_tokens'] = self.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                attention_data['output_tokens'] = self.model.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
            
            return summary, attention_data
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}", {}
    
    def render_attention_visualization(self, attention_data: Dict):
        """Render attention heatmap"""
        if not attention_data or 'weights' not in attention_data:
            st.warning("No attention data available for visualization.")
            return
        
        st.subheader("🔍 Attention Visualization")
        
        weights = attention_data['weights']
        input_tokens = attention_data['input_tokens'][:50]  # Limit for display
        output_tokens = attention_data['output_tokens'][:20]  # Limit for display
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=weights[:len(output_tokens), :len(input_tokens)],
            x=input_tokens,
            y=output_tokens,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Attention Weights (Output vs Input Tokens)",
            xaxis_title="Input Tokens",
            yaxis_title="Output Tokens",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_metrics_analysis(self, original_text: str, summary: str):
        """Render metrics analysis"""
        st.subheader("📊 Summary Analysis")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        with col1:
            st.metric("Original Length", f"{original_words} words")
        with col2:
            st.metric("Summary Length", f"{summary_words} words")
        with col3:
            st.metric("Compression Ratio", f"{compression_ratio:.2f}")
        with col4:
            reading_time = summary_words / 200  # Assume 200 WPM
            st.metric("Reading Time", f"{reading_time:.1f} min")
        
        # Quality metrics (simplified)
        try:
            metrics = self.evaluator.compute_metrics([summary], [original_text])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Content Coverage", f"{metrics.get('content_coverage', 0):.3f}")
            with col2:
                st.metric("Repetition Score", f"{metrics.get('repetition_score', 0):.3f}")
            with col3:
                st.metric("Novelty Score", f"{metrics.get('novelty_score', 0):.3f}")
                
        except Exception as e:
            st.warning(f"Could not compute detailed metrics: {e}")
    
    def render_examples(self):
        """Render example texts"""
        st.subheader("📝 Example Articles")
        
        examples = [
            {
                "title": "Technology News",
                "text": """Công ty công nghệ hàng đầu Việt Nam vừa công bố kết quả kinh doanh quý III/2024 với doanh thu đạt 1.2 tỷ USD, tăng 15% so với cùng kỳ năm trước. Điều này cho thấy sự phục hồi mạnh mẽ của ngành công nghệ sau giai đoạn khó khăn. Công ty cũng thông báo kế hoạch mở rộng hoạt động sang thị trường Đông Nam Á trong năm 2025, với mục tiêu tăng trưởng 25% doanh thu. Các chuyên gia đánh giá đây là bước đi chiến lược quan trọng giúp công ty củng cố vị thế trong khu vực."""
            },
            {
                "title": "Economic News",
                "text": """Ngân hàng Nhà nước Việt Nam quyết định giữ nguyên lãi suất điều hành ở mức 4.5% trong phiên họp tháng 11/2024. Quyết định này được đưa ra sau khi xem xét các yếu tố kinh tế vĩ mô, bao gồm tỷ lệ lạm phát đang được kiểm soát tốt ở mức 3.2% và tăng trưởng GDP quý III đạt 6.8%. Các chuyên gia kinh tế cho rằng việc duy trì lãi suất ổn định sẽ hỗ trợ doanh nghiệp tiếp cận vốn và thúc đẩy đầu tư trong bối cảnh kinh tế toàn cầu còn nhiều bất ổn."""
            }
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Load Example {i+1}: {example['title']}", key=f"example_{i}"):
                st.session_state.input_text = example['text']
                st.experimental_rerun()
    
    def run(self):
        """Main application runner"""
        # Initialize components
        if not self.initialize_components():
            st.stop()
        
        # Render UI
        self.render_header()
        generation_params, enhancements = self.render_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col2:
            self.render_examples()
        
        with col1:
            st.subheader("📰 Input Article")
            
            # Text input
            input_text = st.text_area(
                "Enter Vietnamese news article:",
                value=st.session_state.get('input_text', ''),
                height=200,
                placeholder="Nhập bài báo tiếng Việt tại đây..."
            )
            
            # Generate button
            if st.button("🚀 Generate Summary", type="primary"):
                if input_text.strip():
                    with st.spinner("Generating summary..."):
                        summary, attention_data = self.generate_summary(
                            input_text, generation_params, enhancements
                        )
                    
                    # Display results
                    st.subheader("📋 Generated Summary")
                    st.write(summary)
                    
                    # Metrics analysis
                    self.render_metrics_analysis(input_text, summary)
                    
                    # Attention visualization
                    if enhancements['use_attention_viz']:
                        self.render_attention_visualization(attention_data)
                    
                    # Save to session state
                    st.session_state.last_summary = summary
                    st.session_state.last_input = input_text
                    
                else:
                    st.warning("Please enter some text to summarize.")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🎓 **Enhanced Vietnamese Summarization** - Graduation Thesis Project | "
            "Built with Streamlit, PyTorch, and Transformers"
        )


def main():
    """Main function"""
    demo = SummarizationDemo()
    demo.run()


if __name__ == "__main__":
    main()
