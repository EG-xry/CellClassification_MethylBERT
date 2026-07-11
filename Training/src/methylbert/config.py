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
    loss="cross_entropy"  # Changed default to cross_entropy for multi-class
    num_labels=-1
    num_classes=39  # Default for multi-class classification

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
            ('save_freq', None),  # How often to save model checkpoints
            
            # Model architecture
            ('n_hidden', None),  # Number of hidden layers (if specified)
            ('num_classes', 39),  # Number of classes for multi-class classification
            
            # Training control
            ('decrease_steps', 200),  # Steps for learning rate decay
            ('eval', False),  # Whether to run evaluation only
            ('gradient_accumulation_steps', 1),  # Gradient accumulation for large batches
            ('max_grad_norm', 1.0),  # Maximum gradient norm for clipping
            
            # Loss function
            ('loss', 'cross_entropy')  # Loss function: 'bce', 'focal_bce', or 'cross_entropy'
          ]
        )

    if kwargs is not None:
        for key in config.keys():
            if key in kwargs.keys():
                config[key] = kwargs.pop(key)

    return Config(config)