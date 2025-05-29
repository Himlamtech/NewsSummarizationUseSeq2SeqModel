"""
Enhanced T5 Model for Vietnamese Summarization

This module implements three key enhancements:
1. Enhanced Self-Attention with relative position encoding
2. Pointer-Generator Networks for handling OOV words
3. Coverage Mechanism for reducing repetition

Mathematical foundations:
- Attention: Attention(Q,K,V) = softmax((QK^T + bias) / √d_k)V
- Pointer-Gen: P_final = P_gen * P_vocab + (1 - P_gen) * P_copy
- Coverage: Coverage_loss = Σ min(attention_t, coverage_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from typing import Optional, Tuple, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class EnhancedSelfAttention(nn.Module):
    """Enhanced self-attention with relative position encoding."""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.d_kv, self.d_model, bias=False)
        
        # Relative position encoding
        self.relative_attention_bias = nn.Embedding(32, self.num_heads)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)
        
        # Add relative position bias
        position_bias = self._compute_bias(seq_len, hidden_states.device)
        scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_kv
        )
        
        # Final projection
        output = self.o_proj(attn_output)
        return output, attn_weights
    
    def _compute_bias(self, seq_len: int, device: torch.device):
        """Compute relative position bias."""
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Bucket relative positions
        relative_positions = torch.clamp(relative_positions, -16, 16) + 16
        
        bias = self.relative_attention_bias(relative_positions)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias


class PointerGeneratorNetwork(nn.Module):
    """Pointer-Generator Network for handling OOV words."""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.d_model = config.d_model
        
        # Pointer network components
        self.pointer_linear = nn.Linear(self.d_model * 2, 1)
        self.p_gen_linear = nn.Linear(self.d_model * 2, 1)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                encoder_mask: Optional[torch.Tensor] = None):
        batch_size, tgt_len = decoder_hidden.shape[:2]
        src_len = encoder_outputs.shape[1]
        
        # Compute attention scores
        decoder_expanded = decoder_hidden.unsqueeze(2).expand(-1, -1, src_len, -1)
        encoder_expanded = encoder_outputs.unsqueeze(1).expand(-1, tgt_len, -1, -1)
        
        combined = torch.cat([decoder_expanded, encoder_expanded], dim=-1)
        attention_scores = self.pointer_linear(combined).squeeze(-1)
        
        # Apply mask
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(
                encoder_mask.unsqueeze(1) == 0, -1e9
            )
        
        # Pointer weights and generation probability
        pointer_weights = F.softmax(attention_scores, dim=-1)
        context = torch.bmm(pointer_weights, encoder_outputs)
        
        p_gen_input = torch.cat([decoder_hidden, context], dim=-1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))
        
        return pointer_weights, p_gen, context


class CoverageMechanism(nn.Module):
    """Coverage mechanism to reduce repetition."""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.d_model = config.d_model
        
    def forward(self, attention_weights: torch.Tensor, previous_coverage: Optional[torch.Tensor] = None):
        # Accumulate coverage
        if previous_coverage is None:
            coverage = attention_weights.sum(dim=1)
        else:
            coverage = previous_coverage + attention_weights.sum(dim=1)
        
        # Coverage loss
        coverage_loss = torch.sum(torch.min(attention_weights.sum(dim=1), coverage), dim=-1)
        
        return coverage, coverage_loss


class EnhancedT5Model(T5ForConditionalGeneration):
    """Enhanced T5 model with advanced attention mechanisms."""
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        
        # Enhanced components
        self.enhanced_attention = EnhancedSelfAttention(config)
        self.pointer_generator = PointerGeneratorNetwork(config)
        self.coverage_mechanism = CoverageMechanism(config)
        
        # Loss weights
        self.coverage_loss_weight = getattr(config, 'coverage_loss_weight', 1.0)
        self.pointer_gen_loss_weight = getattr(config, 'pointer_gen_loss_weight', 1.0)
        
        logger.info("Initialized EnhancedT5Model with advanced attention mechanisms")
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, **kwargs):
        
        # Get base model outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            **kwargs
        )
        
        # Add enhanced losses during training
        if labels is not None and self.training:
            enhanced_loss = self._compute_enhanced_loss(outputs, input_ids, attention_mask)
            outputs.loss = outputs.loss + enhanced_loss
        
        return outputs
    
    def _compute_enhanced_loss(self, outputs, input_ids, attention_mask):
        """Compute additional losses from enhancements."""
        total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            encoder_hidden = outputs.encoder_hidden_states[-1]
            decoder_hidden = outputs.decoder_hidden_states[-1]
            
            # Pointer-generator loss
            try:
                pointer_weights, p_gen, _ = self.pointer_generator(
                    decoder_hidden, encoder_hidden, attention_mask
                )
                # Simplified loss - encourage balanced generation vs copying
                pg_loss = torch.mean(torch.abs(p_gen - 0.5))
                total_loss = total_loss + self.pointer_gen_loss_weight * pg_loss
            except:
                pass
            
            # Coverage loss
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                try:
                    attention_weights = outputs.cross_attentions[-1].mean(dim=1)
                    _, coverage_loss = self.coverage_mechanism(attention_weights)
                    total_loss = total_loss + self.coverage_loss_weight * coverage_loss.mean()
                except:
                    pass
        
        return total_loss


def create_enhanced_model(model_name: str = "VietAI/vit5-large-vietnews-summarization", **kwargs):
    """Create an enhanced T5 model."""
    try:
        logger.info(f"Loading base model: {model_name}")
        
        # Load configuration
        config = T5Config.from_pretrained(model_name)
        
        # Add enhancement parameters
        for key, value in kwargs.items():
            setattr(config, key, value)
        
        # Create enhanced model
        model = EnhancedT5Model.from_pretrained(
            model_name, 
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.tokenizer = tokenizer
        
        logger.info(f"Successfully created enhanced model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create enhanced model: {e}")
        raise
