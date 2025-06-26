"""DomainBERT model implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import (
    PreTrainedModel,
    AutoModel,
    BertModel,
    BertConfig
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
    MaskedLMOutput,
)
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertPooler
)

from .config import DomainBertConfig


class DomainEmbeddings(nn.Module):
    """Embeddings for DomainBERT with TLD integration"""
    
    def __init__(self, config: DomainBertConfig):
        super().__init__()
        self.char_embeddings = nn.Embedding(config.char_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
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
            vocab_size=config.char_vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act="gelu",
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
    """DomainBERT for masked language modeling with multi-task TLD prediction"""
    
    config_class = DomainBertConfig
    base_model_prefix = "domain_bert"
    
    def __init__(self, config: DomainBertConfig):
        super().__init__(config)
        self.config = config
        
        # Base model
        self.domain_bert = DomainBertModel(config)
        
        # MLM head
        self.mlm_predictions = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.char_vocab_size)
        )
        
        # TLD prediction head
        if config.use_tld_embeddings:
            self.tld_classifier = nn.Linear(config.hidden_size, config.tld_vocab_size)
        else:
            self.tld_classifier = None
        
        # Initialize weights
        self.post_init()
    
    def get_output_embeddings(self):
        return self.mlm_predictions[-1]
    
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
        
        # MLM predictions
        mlm_logits = self.mlm_predictions(sequence_output)
        
        # TLD predictions (using pooled output)
        tld_logits = None
        if self.tld_classifier is not None:
            tld_logits = self.tld_classifier(pooled_output)
        
        # Calculate losses
        total_loss = None
        mlm_loss = None
        tld_loss = None
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.char_vocab_size), labels.view(-1))
            total_loss = self.config.mlm_weight * mlm_loss
        
        if tld_labels is not None and tld_logits is not None:
            loss_fct = nn.CrossEntropyLoss()
            tld_loss = loss_fct(tld_logits.view(-1, self.config.tld_vocab_size), tld_labels.view(-1))
            if total_loss is None:
                total_loss = self.config.tld_weight * tld_loss
            else:
                total_loss = total_loss + self.config.tld_weight * tld_loss
        
        if not return_dict:
            output = (mlm_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DomainBertForSequenceClassification(PreTrainedModel):
    """DomainBERT for sequence classification tasks"""
    
    config_class = DomainBertConfig
    base_model_prefix = "domain_bert"
    
    def __init__(self, config: DomainBertConfig, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config
        
        # Base model
        self.domain_bert = DomainBertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        tld_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
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
        
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )