import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import math, os
from copy import deepcopy

from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from methylbert.function import FocalLoss, DecoupledFocalLoss
from methylbert.config import MethylBERTConfig

METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/resolve/main/pytorch_model.bin",
}

class MethylBertEmbeddedDMR(BertPreTrainedModel):
    pretrained_model_archive_map = METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    config_class = MethylBERTConfig
    base_model_prefix = "methylbert"

    def __init__(self, config, seq_len=150, num_classes=None, class_weights: torch.Tensor = None, **kwargs):
        # from pretrained - calls the init
        super().__init__(config)
        
        # Extract additional parameters from kwargs with proper handling
        num_labels = kwargs.get('num_labels', None)
        num_dmr_embeddings = kwargs.get('num_dmr_embeddings', None) 
        loss = kwargs.get('loss', None)
        
        # Validate and set seq_len - ensure it's not None
        if seq_len is None:
            seq_len = getattr(config, 'seq_len', 150)
            print(f"seq_len was None, using config or default: {seq_len}")
        else:
            print(f"Using explicit seq_len: {seq_len}")
        
        # Ensure seq_len is a valid positive integer
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len}")
        
        # Following HuggingFace convention: num_labels = number of classification classes
        # Use num_classes from config if not provided explicitly, with fallback hierarchy
        if num_classes is None:
            # Priority: explicit num_labels > config.num_classes > config.num_labels > default
            self.num_classes = (num_labels or 
                               getattr(config, 'num_classes', None) or 
                               getattr(config, 'num_labels', 39))
            print(f"Using num_classes from hierarchy: {self.num_classes}")
        else:
            self.num_classes = num_classes
            print(f"Using explicit num_classes: {self.num_classes}")
        
        # Set num_labels for HuggingFace compatibility (classification head size)
        self.num_labels = self.num_classes
        
        # DMR embedding size - separate from classification
        # Priority: explicit num_dmr_embeddings > config.num_dmr_embeddings > config.vocab_size > default
        self.num_dmr_embeddings = (num_dmr_embeddings or 
                                  getattr(config, 'num_dmr_embeddings', None) or 
                                  getattr(config, 'vocab_size', 30000))
        print(f"DMR embedding size: {self.num_dmr_embeddings}")
        
        # Validate DMR embedding size
        if self.num_dmr_embeddings is None or not isinstance(self.num_dmr_embeddings, int) or self.num_dmr_embeddings <= 0:
            raise ValueError(f"num_dmr_embeddings must be a positive integer, got {self.num_dmr_embeddings}")

        # Determine loss function from kwargs or config
        self.loss = loss or getattr(config, 'loss', None)
        
        # Validate loss
        if self.loss is None:
            raise ValueError(f"loss is missing. Provide via kwargs or config. config has attributes: {list(vars(config).keys())}")
        
        if self.loss not in ["bce", "focal", "cross_entropy", "decoupled_focal"]:
            raise ValueError(f"loss must be bce, focal, cross_entropy, or decoupled_focal. {self.loss} is given.")
        self.class_weights = class_weights  # optional class weights for cross-entropy
        self.classification_loss_fct = self._setup_loss(self.loss, verbose=False)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.read_classifier = nn.Sequential(
            nn.Linear((config.hidden_size+1)*(seq_len+1), seq_len+1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.ReLU(),
            nn.LayerNorm(seq_len+1, eps=config.layer_norm_eps),
            nn.Linear(seq_len+1, self.num_classes)
        )

        self.seq_len = seq_len

        # Create embedding layer without device/dtype to avoid None values
        self.dmr_encoder = nn.Sequential(
            nn.Embedding(self.num_dmr_embeddings, seq_len + 1),
        )
        
        # Call init_weights to handle device/dtype properly
        self.init_weights()

    def _setup_loss(self, loss, verbose=True):
        if loss == "bce":
            if verbose:
                print("Binary Cross Entropy loss assigned")
            return nn.BCEWithLogitsLoss() # for binary classification
        elif loss == "focal":
            if verbose:
                print("Focal loss assigned (softmax-based multiclass)")
            # Get per-class weights (equivalent to old alpha but using weight parameter)
            weight = getattr(self.config, "focal_weight", None)
            gamma = getattr(self.config, "focal_gamma", 2.0)
            
            # Convert weight to tensor and move to model device if provided
            if weight is not None:
                if isinstance(weight, (list, tuple)):
                    # Get model device to ensure weights are on the same device
                    # Handle case where model might not be on device yet
                    try:
                        device = next(self.parameters()).device
                    except (StopIteration, RuntimeError):
                        # Model not on device yet, use CPU for now
                        device = torch.device('cpu')
                        if verbose:
                            print("Warning: Model not on device yet, using CPU for focal weights")
                    
                    weight = torch.tensor(weight, dtype=torch.float32, device=device)
                    if verbose:
                        print(f"Focal weight (per-class) length={len(weight)}; values: {weight.detach().cpu().tolist()}")
                elif isinstance(weight, torch.Tensor):
                    # Ensure tensor is on the same device as model
                    try:
                        device = next(self.parameters()).device
                    except (StopIteration, RuntimeError):
                        # Model not on device yet, use CPU for now
                        device = torch.device('cpu')
                        if verbose:
                            print("Warning: Model not on device yet, using CPU for focal weights")
                    
                    weight = weight.to(device)
                    if verbose:
                        print(f"Focal weight (per-class) length={len(weight)}; values: {weight.detach().cpu().tolist()}")
                else:
                    if verbose:
                        print(f"Focal weight: {weight}")
            else:
                if verbose:
                    print("Focal weight: None (equal class weighting)")
            
            if verbose:
                print(f"Focal gamma: {gamma}")
            return FocalLoss(weight=weight, gamma=gamma)
        elif loss == "decoupled_focal":
            if verbose:
                print("Decoupled Focal loss assigned")
            alpha_pos = getattr(self.config, "focal_alpha_pos", None)
            alpha_neg = getattr(self.config, "focal_alpha_neg", None)
            gamma = getattr(self.config, "focal_gamma", 2.0)
            
            # Print resolved alpha values
            if verbose:
                if isinstance(alpha_pos, torch.Tensor):
                    alpha_pos_list = alpha_pos.detach().cpu().tolist()
                    print(f"Focal alpha_pos (per-class) length={len(alpha_pos_list)}; values: {alpha_pos_list}")
                elif isinstance(alpha_pos, (list, tuple)):
                    alpha_pos_list = list(alpha_pos)
                    print(f"Focal alpha_pos (per-class) length={len(alpha_pos_list)}; values: {alpha_pos_list}")
                else:
                    print(f"Focal alpha_pos: {alpha_pos}")
                    
                if isinstance(alpha_neg, torch.Tensor):
                    alpha_neg_list = alpha_neg.detach().cpu().tolist()
                    print(f"Focal alpha_neg (per-class) length={len(alpha_neg_list)}; values: {alpha_neg_list}")
                elif isinstance(alpha_neg, (list, tuple)):
                    alpha_neg_list = list(alpha_neg)
                    print(f"Focal alpha_neg (per-class) length={len(alpha_neg_list)}; values: {alpha_neg_list}")
                else:
                    print(f"Focal alpha_neg: {alpha_neg}")
                
            return DecoupledFocalLoss(alpha_pos=alpha_pos, alpha_neg=alpha_neg, gamma=gamma)
        elif loss == "cross_entropy":
            # Attach class weights if provided (for imbalanced multi-class training)
            label_smoothing = getattr(self.config, "label_smoothing", 0.0)
            if self.class_weights is not None:
                # ensure weights are on the same device as model parameters
                try:
                    device = next(self.parameters()).device
                except (StopIteration, RuntimeError):
                    # Model not on device yet, use CPU for now
                    device = torch.device('cpu')
                    if verbose:
                        print("Warning: Model not on device yet, using CPU for class weights")
                
                weights = self.class_weights.to(device)
                if verbose:
                    print("Cross Entropy loss assigned with class weights")
                return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            else:
                if verbose:
                    print("Cross Entropy loss assigned for multi-class classification")
                return nn.CrossEntropyLoss(label_smoothing=label_smoothing) # for multi-class classification

    def set_class_weights(self, weights: torch.Tensor):
        """
        Set or update class weights for cross-entropy loss. The average weight is
        expected to be around 1. This will rebuild the loss function accordingly.
        """
        self.class_weights = weights
        if self.loss == "cross_entropy":
            try:
                device = next(self.parameters()).device
            except (StopIteration, RuntimeError):
                # Model not on device yet, use CPU for now
                device = torch.device('cpu')
                print("Warning: Model not on device yet, using CPU for class weights in set_class_weights")
            
            self.classification_loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(device),
                label_smoothing=getattr(self.config, "label_smoothing", 0.0)
            )

    def set_dropout(self, p: float):
        """Update dropout probability across the model (BERT + heads)."""
        self.config.hidden_dropout_prob = p
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def set_label_smoothing(self, epsilon: float):
        """Update label smoothing and rebuild the CE loss accordingly."""
        self.config.label_smoothing = float(epsilon)
        if self.loss == "cross_entropy":
            try:
                device = next(self.parameters()).device
            except (StopIteration, RuntimeError):
                # Model not on device yet, use CPU for now
                device = torch.device('cpu')
                print("Warning: Model not on device yet, using CPU for class weights in set_label_smoothing")
            
            weight = self.class_weights.to(device) if self.class_weights is not None else None
            self.classification_loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.config.label_smoothing)

    def check_model_status(self):
        print("Bert model training mode : %s"%(self.bert.training))
        print("Dropout training mode : %s"%(self.dropout.training))
        print("Read classifier training mode : %s"%(self.read_classifier.training))
        
    def from_pretrained_read_classifier(self, pretrained_model_name_or_path, device="cpu"):
        self.read_classifier.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def from_pretrained_dmr_encoder(self, pretrained_model_name_or_path, device="cpu"):
        self.dmr_encoder.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def forward(
        self,
        step,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ctype_label=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        #DMR info 
        encoded_dmr = self.dmr_encoder(labels.view(-1))

        sequence_output =  torch.cat((sequence_output, encoded_dmr.unsqueeze(-1)), axis=-1)

        ctype_logits = self.read_classifier(sequence_output.view(-1,(self.seq_len+1)*769))
        
        # Losses
        if self.loss == "bce":
            # Binary classification - convert to one-hot and use BCE
            loss = self.classification_loss_fct(ctype_logits.view(-1, self.num_classes), 
                                                F.one_hot(ctype_label, num_classes=self.num_classes).to(torch.float32).view(-1, self.num_classes))
            probs = ctype_logits.softmax(dim=1)
        elif self.loss == "focal":
            # Softmax-based focal loss uses class indices directly
            loss = self.classification_loss_fct(ctype_logits.view(-1, self.num_classes), ctype_label.view(-1))
            probs = ctype_logits.softmax(dim=1)
        elif self.loss == "decoupled_focal":
            # Decoupled focal loss uses class indices (automatically converts to one-hot internally)
            loss = self.classification_loss_fct(ctype_logits.view(-1, self.num_classes), ctype_label.view(-1))
            probs = torch.sigmoid(ctype_logits)  # Decoupled focal uses sigmoid probabilities
        else:
            # Multi-class classification - use CrossEntropyLoss directly
            loss = self.classification_loss_fct(ctype_logits.view(-1, self.num_classes), ctype_label.view(-1))
            probs = ctype_logits.softmax(dim=1)
        
        outputs = {"loss": loss,
                    "dmr_logits":sequence_output,
                   "classification_logits": probs}
        
        return outputs  # (loss), logits, (hidden_states), (attentions)


