from collections import OrderedDict

from transformers import BertConfig

METHYLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/raw/main/config.json",
    "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/raw/main/config.json",
    "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/raw/main/config.json",
    "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/raw/main/config.json",
    "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/raw/main/config.json"
}

class MethylBERTConfig(BertConfig):
    pretrained_config_archive_map = METHYLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    loss = "decoupled_focal" 
    num_labels = None  # Number of classification classes (HuggingFace convention)
    num_classes = None  # Will be auto-detected from dataset
    num_dmr_embeddings = None  # Number of DMR embeddings (auto-detected from dataset)
    label_smoothing = 0.0  # Default label smoothing for cross-entropy
    focal_weight = None  # Default None -> auto-compute per-class weights from dataset  
    focal_gamma = 2.0  # Default gamma for focal loss
    # New decoupled focal loss parameters
    use_decoupled_focal = True  # Whether to use decoupled focal loss
    focal_alpha_pos = None  # Positive example weights (auto-computed if None)
    focal_alpha_neg = None  # Negative example weights (auto-computed if None)

class Config(object):
    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def get_config(**kwargs):
    '''
    Create a Config object for configuration from input
    '''
    config = OrderedDict(
          [
            # Training parameters
            ('lr', 1e-4),  # Learning rate
            ('beta', (0.9, 0.999)),  # Adam optimizer betas
            ('weight_decay', 0.01),  # Weight decay for regularization
            ('warmup_step', 10000),  # Learning rate warmup steps
            ('eps', 1e-6),  # Epsilon for Adam optimizer
            
            # Hardware and device settings
            ('with_cuda', True),  # Use GPU if available
            ('amp', False),  # Automatic mixed precision
            
            # Logging and evaluation
            ('log_freq', 10),  # How often to log training progress
            ('eval_freq', 1),  # How often to evaluate on validation set
            
            # Model architecture
            ('n_hidden', None),  # Number of hidden layers (if specified)
            ('num_classes', None),  # Number of classes for multi-class classification (auto-detected from dataset)
            ('dropout', 0.1),  # Global dropout probability for model
            
            # Training control
            ('decrease_steps', 200),  # Steps for learning rate decay
            ('eval', False),  # Whether to run evaluation only
            ('gradient_accumulation_steps', 1),  # Gradient accumulation for large batches
            ('max_grad_norm', 1.0),  # Maximum gradient norm for clipping
            
            # Balanced sampling control
            ('use_balanced_batches', True),  # Whether to use balanced batch sampling for training
            ('use_balanced_test_eval', True),  # Whether to use balanced sampling for test evaluation
            
            # Loss function (default to focal loss) and related params
            ('loss', 'decoupled_focal'),  # 'focal', 'bce', 'cross_entropy', or 'decoupled_focal'
            ('label_smoothing', 0.0),  # Epsilon for label smoothing with cross-entropy
            ('focal_weight', None),  # None -> auto-compute per-class weights from dataset, or provide list to override
            ('focal_gamma', 2.0),  # Gamma parameter for focal loss
            # Decoupled focal loss parameters
            ('use_decoupled_focal', True),  # Whether to use decoupled focal loss
            ('focal_alpha_pos', None),  # Positive example weights (auto-computed if None)
            ('focal_alpha_neg', None),  # Negative example weights (auto-computed if None)
          ]
        )

    if kwargs is not None:
        for key in config.keys():
            if key in kwargs.keys():
                config[key] = kwargs.pop(key)

    return Config(config)