"""DomainBERT model v2 with separate char/TLD prediction heads"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict
from transformers import (
    PreTrainedModel,
    BertModel,
    BertConfig
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler
)

from .config import DomainBertConfig
# Import DomainEmbeddings from same file below


class DomainEmbeddings(nn.Module):
    """Embeddings for DomainBERT with TLD integration"""
    
    def __init__(self, config: DomainBertConfig):
        super().__init__()
        # Character embeddings for domain characters
        self.char_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # TLD embeddings
        if config.use_tld_embeddings:
            self.tld_embeddings = nn.Embedding(config.tld_vocab_size, config.tld_embed_dim)
            self.tld_projection = nn.Linear(config.tld_embed_dim, config.hidden_size)
        else:
            self.tld_embeddings = None
            self.tld_projection = None
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs buffer
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        tld_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.char_embeddings(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Position embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        
        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        # Add TLD information if available
        if self.tld_embeddings is not None and tld_ids is not None:
            # Get TLD embeddings
            tld_embeds = self.tld_embeddings(tld_ids)  # [batch_size, tld_embed_dim]
            tld_embeds = self.tld_projection(tld_embeds)  # [batch_size, hidden_size]
            
            # Broadcast TLD embeddings across sequence
            tld_embeds = tld_embeds.unsqueeze(1).expand(-1, seq_length, -1)
            
            # Add to embeddings
            embeddings = embeddings + tld_embeds
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class DomainBertModel(PreTrainedModel):
    """Base DomainBERT model"""
    
    config_class = DomainBertConfig
    base_model_prefix = "domain_bert"
    
    def __init__(self, config: DomainBertConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = DomainEmbeddings(config)
        
        # Create BERT encoder with our config
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
        )
        
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        
        # Initialize weights
        self.post_init()
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.encoder.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        return self.embeddings.char_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.char_embeddings = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        tld_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            tld_ids=tld_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # Prepare attention mask for encoder
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # Encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DomainBertForMaskedLM(PreTrainedModel):
    """DomainBERT for masked language modeling with separate char/TLD heads"""
    
    config_class = DomainBertConfig
    base_model_prefix = "domain_bert"
    
    def __init__(self, config: DomainBertConfig):
        super().__init__(config)
        self.config = config
        
        # Base model
        self.domain_bert = DomainBertModel(config)
        
        # Character prediction head (for positions 0-43)
        self.char_predictions = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, 44)  # Character vocabulary including special tokens
        )
        
        # TLD prediction head (for positions 43-554)
        self.tld_predictions = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, 511)  # Only TLD vocabulary
        )
        
        # TLD classifier for CLS token (global TLD prediction)
        self.cls_tld_classifier = nn.Linear(config.hidden_size, 511)
        
        # Initialize weights
        self.post_init()
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.domain_bert.encoder, 'gradient_checkpointing_enable'):
            self.domain_bert.encoder.gradient_checkpointing_enable()
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self.domain_bert.encoder, 'gradient_checkpointing_disable'):
            self.domain_bert.encoder.gradient_checkpointing_disable()
    
    def create_position_masks(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Create masks to identify character vs TLD positions
        
        When labels are provided, use them to determine the original token type
        for masked positions.
        """
        # Start with basic classification
        is_char_position = torch.zeros_like(input_ids, dtype=torch.bool)
        is_tld_position = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # For each position, determine if it's char or TLD
        for i in range(input_ids.shape[0]):  # batch dimension
            for j in range(input_ids.shape[1]):  # sequence dimension
                token_id = input_ids[i, j].item()
                
                # If we have labels and this is a masked position, use the label
                if labels is not None and token_id == self.config.mask_token_id and labels[i, j] != -100:
                    original_token = labels[i, j].item()
                    if original_token < 44:
                        is_char_position[i, j] = True
                    else:
                        is_tld_position[i, j] = True
                # Otherwise classify based on the actual token
                elif token_id not in [self.config.pad_token_id, self.config.cls_token_id, 
                                     self.config.sep_token_id, self.config.unk_token_id]:
                    if token_id < 44:  # Character or dot
                        is_char_position[i, j] = True
                    else:  # TLD token
                        is_tld_position[i, j] = True
        
        return {
            "is_char_position": is_char_position,
            "is_tld_position": is_tld_position
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        tld_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        tld_labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through base model
        outputs = self.domain_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            tld_ids=tld_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        pooled_output = outputs[1]    # [batch_size, hidden_size]
        
        # Get position masks (pass labels to handle masked positions correctly)
        position_masks = self.create_position_masks(input_ids, labels)
        is_char_position = position_masks["is_char_position"]
        is_tld_position = position_masks["is_tld_position"]
        
        # Get predictions from both heads
        char_logits = self.char_predictions(sequence_output)  # [batch_size, seq_length, 44]
        tld_logits = self.tld_predictions(sequence_output)    # [batch_size, seq_length, 511]
        
        # CLS TLD prediction
        cls_tld_logits = self.cls_tld_classifier(pooled_output)  # [batch_size, 511]
        
        # Calculate losses
        total_loss = None
        char_loss = None
        tld_loss = None
        cls_loss = None
        
        if labels is not None:
            # Split labels by position type
            char_labels = labels.clone()
            tld_labels_seq = labels.clone()
            
            # For character positions: set TLD positions to -100
            char_labels[is_tld_position] = -100
            
            # For TLD positions: set char positions to -100 and adjust IDs
            tld_labels_seq[is_char_position] = -100
            # Adjust TLD token IDs (44-554) to TLD vocabulary indices (0-510)
            # Note: token 43 is '.', actual TLD tokens start at 44
            tld_mask = tld_labels_seq != -100
            if tld_mask.any():
                # Adjust TLD token IDs, but handle dots specially
                # Dots (43) are marked as TLD positions but shouldn't be predicted
                tld_labels_seq[tld_mask & (tld_labels_seq == 43)] = -100
                # Adjust actual TLD tokens (44+) to 0-based indices
                actual_tld_mask = tld_mask & (tld_labels_seq >= 44)
                tld_labels_seq[actual_tld_mask] = tld_labels_seq[actual_tld_mask] - 44
            
            # Calculate character loss
            if (char_labels != -100).any():
                loss_fct = nn.CrossEntropyLoss()
                char_loss = loss_fct(char_logits.view(-1, 44), char_labels.view(-1))
                total_loss = char_loss
            
            # Calculate TLD sequence loss
            if (tld_labels_seq != -100).any():
                loss_fct = nn.CrossEntropyLoss()
                tld_loss = loss_fct(tld_logits.view(-1, 511), tld_labels_seq.view(-1))
                if total_loss is None:
                    total_loss = 0.5 * tld_loss
                else:
                    total_loss = total_loss + 0.5 * tld_loss
        
        # CLS TLD prediction loss
        if tld_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(cls_tld_logits.view(-1, 511), tld_labels.view(-1))
            if total_loss is None:
                total_loss = 0.5 * cls_loss
            else:
                total_loss = total_loss + 0.5 * cls_loss
        
        # Combine logits for compatibility (pad smaller vocab)
        # This is just for interface compatibility - the actual predictions use separate heads
        char_logits_padded = F.pad(char_logits, (0, 511), value=-float('inf'))
        tld_logits_padded = F.pad(tld_logits, (44, 0), value=-float('inf'))
        
        # Use character logits for char positions, TLD logits for TLD positions
        combined_logits = torch.where(
            is_char_position.unsqueeze(-1),
            char_logits_padded,
            tld_logits_padded
        )
        
        if not return_dict:
            output = (combined_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=combined_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )