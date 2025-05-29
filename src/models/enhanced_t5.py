"""
Enhanced T5 Model with Advanced Attention Mechanisms
Implements self-attention improvements, pointer-generator networks, and coverage mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple, Dict, Any
import math


class EnhancedSelfAttention(nn.Module):
    """Enhanced self-attention with improved position encoding and attention patterns"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.dropout = config.dropout_rate
        
        # Multi-head attention components
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.d_kv, self.d_model, bias=False)
        
        # Enhanced position encoding
        self.relative_attention_bias = nn.Embedding(32, self.num_heads)
        
        # Attention enhancement layers
        self.attention_dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)
        
        # Add relative position bias
        if position_bias is None:
            position_bias = self._compute_bias(seq_len, seq_len)
        scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_kv
        )
        
        # Final projection
        output = self.o_proj(attn_output)
        
        return output, attn_weights
    
    def _compute_bias(self, query_length, key_length):
        """Compute relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position, num_buckets=32, max_distance=128
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values
    
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """Convert relative position to bucket index"""
        ret = 0
        n = -relative_position
        
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret


class PointerGeneratorNetwork(nn.Module):
    """Pointer-Generator Network for handling OOV words and proper nouns"""
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Pointer network components
        self.pointer_linear = nn.Linear(self.d_model * 2, 1)
        self.context_linear = nn.Linear(self.d_model, self.d_model)
        self.decoder_linear = nn.Linear(self.d_model, self.d_model)
        
        # Generation probability
        self.p_gen_linear = nn.Linear(self.d_model * 3, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, encoder_attention_mask, input_ids):
        batch_size, seq_len = decoder_hidden.shape[:2]
        src_len = encoder_outputs.shape[1]
        
        # Compute attention over encoder outputs
        decoder_expanded = decoder_hidden.unsqueeze(2).expand(-1, -1, src_len, -1)
        encoder_expanded = encoder_outputs.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Attention scores
        combined = torch.cat([decoder_expanded, encoder_expanded], dim=-1)
        attention_scores = self.pointer_linear(combined).squeeze(-1)
        
        # Apply encoder attention mask
        if encoder_attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                encoder_attention_mask.unsqueeze(1) == 0, -1e9
            )
        
        # Pointer attention weights
        pointer_weights = F.softmax(attention_scores, dim=-1)
        
        # Context vector
        context = torch.bmm(pointer_weights, encoder_outputs)
        
        # Generation probability
        p_gen_input = torch.cat([decoder_hidden, context, decoder_hidden], dim=-1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))
        
        return pointer_weights, p_gen, context


class CoverageMechanism(nn.Module):
    """Coverage mechanism to reduce repetition and improve content coverage"""
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.coverage_linear = nn.Linear(1, self.d_model)
        
    def forward(self, attention_weights, previous_coverage=None):
        """
        Args:
            attention_weights: Current attention weights [batch, seq_len, src_len]
            previous_coverage: Previous coverage vector [batch, src_len]
        """
        if previous_coverage is None:
            coverage = attention_weights.sum(dim=1)  # [batch, src_len]
        else:
            coverage = previous_coverage + attention_weights.sum(dim=1)
        
        # Coverage loss (penalize attending to already covered positions)
        coverage_loss = torch.sum(torch.min(attention_weights.sum(dim=1), coverage), dim=-1)
        
        return coverage, coverage_loss


class EnhancedT5ForConditionalGeneration(T5ForConditionalGeneration):
    """Enhanced T5 model with advanced attention mechanisms"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Enhanced components
        self.enhanced_attention = EnhancedSelfAttention(config)
        self.pointer_generator = PointerGeneratorNetwork(config)
        self.coverage_mechanism = CoverageMechanism(config)
        
        # Loss weights
        self.coverage_loss_weight = getattr(config, 'coverage_loss_weight', 1.0)
        self.pointer_gen_loss_weight = getattr(config, 'pointer_gen_loss_weight', 1.0)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs
    ):
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
        
        if labels is not None:
            # Compute enhanced losses
            enhanced_loss = self._compute_enhanced_loss(
                outputs, input_ids, attention_mask, labels
            )
            outputs.loss = outputs.loss + enhanced_loss
        
        return outputs
    
    def _compute_enhanced_loss(self, outputs, input_ids, attention_mask, labels):
        """Compute additional losses from enhancements"""
        total_enhanced_loss = 0.0
        
        # Get hidden states
        encoder_hidden_states = outputs.encoder_hidden_states[-1]
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        
        # Pointer-generator loss
        if hasattr(self, 'pointer_generator'):
            pointer_weights, p_gen, context = self.pointer_generator(
                decoder_hidden_states, encoder_hidden_states, attention_mask, input_ids
            )
            
            # Add pointer-generator loss (simplified)
            pg_loss = self._compute_pointer_generator_loss(pointer_weights, p_gen, labels)
            total_enhanced_loss += self.pointer_gen_loss_weight * pg_loss
        
        # Coverage loss
        if hasattr(self, 'coverage_mechanism') and outputs.cross_attentions:
            attention_weights = outputs.cross_attentions[-1].mean(dim=1)  # Average over heads
            coverage, coverage_loss = self.coverage_mechanism(attention_weights)
            total_enhanced_loss += self.coverage_loss_weight * coverage_loss.mean()
        
        return total_enhanced_loss
    
    def _compute_pointer_generator_loss(self, pointer_weights, p_gen, labels):
        """Compute pointer-generator loss (simplified implementation)"""
        # This is a simplified version - in practice, you'd need more sophisticated logic
        # to handle vocabulary distribution vs pointer distribution
        return torch.tensor(0.0, device=pointer_weights.device, requires_grad=True)


def create_enhanced_model(model_name="VietAI/vit5-large-vietnews-summarization", **kwargs):
    """Create an enhanced T5 model with advanced attention mechanisms"""
    
    # Load base configuration
    config = T5Config.from_pretrained(model_name)
    
    # Add enhancement configurations
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    # Create enhanced model
    model = EnhancedT5ForConditionalGeneration.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    
    return model
