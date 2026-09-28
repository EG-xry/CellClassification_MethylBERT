import os
import time
import warnings
import platform
import sys

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, classification_report, f1_score
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (BertConfig, BertForMaskedLM,
                          BertForSequenceClassification)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from methylbert.config import MethylBERTConfig, get_config
from methylbert.data.vocab import MethylVocab
from methylbert.network import MethylBertEmbeddedDMR
from methylbert.utils import get_dna_seq

torch.set_warn_always(False) # one warning per process

def get_device():
    """
    Detect and return the best available device for PyTorch
    Priority: CUDA > MPS (Mac) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_name():
    """
    Return a human-readable device name
    """
    if torch.cuda.is_available():
        return f"CUDA GPU ({torch.cuda.get_device_name()})"
    elif torch.backends.mps.is_available():
        return "Apple Silicon GPU (MPS)"
    else:
        return "CPU"

def learning_rate_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, decrease_steps: int):
    """
    Modified version of get_linear_schedule_with_warmup from transformers
    Learning rate scheduler including warm-up, retaining and decrease

    optimizer: torch.optim.Optimizer
        Optimizer
    num_warmup_steps: int
        Initial steps for linear warm-up
    num_training_steps: int
        Total training steps
    decrease_steps:
        Steps when the learning rate decrease starts
    """

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps: # warm-up
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= decrease_steps: # decrease
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - decrease_steps))
            )
        return 1 # Otherwise, keep the current learning rate

    return LambdaLR(optimizer, lr_lambda, last_epoch = -1)



class MethylBertTrainer(object):
    def __init__(self,
                 vocab_size: int,
                 save_path: str = "",
                 train_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 **kwargs):

        # Setup config
        self._config = get_config(**kwargs)

        # Setup dataloader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        # Auto-detect number of classes from the dataset if available
        if self.train_data and hasattr(self.train_data.dataset, 'num_classes'):
            actual_num_classes = self.train_data.dataset.num_classes()
            print(f"Auto-detected {actual_num_classes} classes from training dataset")
            # Override the config num_classes with the actual number from data
            self._config.num_classes = actual_num_classes

        # Enhanced device detection for Mac and other platforms
        self.device = get_device()
        device_name = get_device_name()
        
        # Update config based on detected device
        if self.device.type == "cuda":
            self._config.with_cuda = True
            self._config.amp = torch.cuda.is_available() and self._config.with_cuda
            print(f"Using {device_name}")
        elif self.device.type == "mps":
            self._config.with_cuda = False  # Not CUDA, but still GPU
            self._config.amp = False  # MPS doesn't support AMP yet
            print(f"Using {device_name}")
            print("Note: Automatic Mixed Precision (AMP) is disabled for MPS")
        else:
            self._config.with_cuda = False
            self._config.amp = False
            print(f"Using {device_name}")
            print("No GPU detected. Training will be slower on CPU.")

        # To save the best model
        self.min_loss = np.inf
        self.best_model_f1_score = 0.0  # Track best Macro-F1 Score instead of loss
        self.best_model_step = 0
        self.best_model_saved_path = None
        
        # To save the lowest loss model
        self.best_model_loss = np.inf  # Track best evaluation loss
        self.best_loss_step = 0
        self.lowest_loss_model_saved_path = None
        
        self.step = 0  # Initialize training step counter
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)
        self.f_train = os.path.join(self.save_path, "train.csv")
        self.f_eval = os.path.join(self.save_path, "eval.csv")

        # Training history for plotting
        self.train_history = {"step": [], "loss": [], "accuracy": [], "lr": []}
        self.eval_history = {"step": [], "loss": [], "accuracy": []}
        
        # Initialize dummy bert attribute to prevent errors before model creation
        self.bert = None

    def save(self, file_path="output/bert_trained.model"):
        '''
        Saving the current BERT model on file_path

        file_path: str
            model output path which gonna be file_path+"ep%d" % epoch
        '''
        if self.bert is None:
            print("Warning: No model to save. Call create_model() first.")
            return
            
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        self.bert.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)

    def _setup_model(self):
        '''
        Load the model to the designated device (CPU, CUDA, or MPS) and create an optimiser

        '''
        self.model = self.bert.to(self.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print("Using %d GPUs for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # Only create optimizer if we're in training mode and have the required config parameters
        if not getattr(self._config, 'eval', False):
            try:
                # Check if we have the required training parameters
                required_params = ['lr', 'beta', 'eps', 'weight_decay']
                missing_params = [param for param in required_params if not hasattr(self._config, param)]
                
                if missing_params:
                    print(f"Warning: Missing training parameters in config: {missing_params}")
                    print("Skipping optimizer creation - this is normal when loading models for analysis")
                else:
                    # Setting the AdamW optimizer with hyper-param
                    self.optim = AdamW(self.model.parameters(),
                                       lr=self._config.lr, betas=self._config.beta, eps=self._config.eps, weight_decay=self._config.weight_decay)
                    print("Optimizer created successfully")
            except Exception as opt_error:
                print(f"Warning: Could not create optimizer: {opt_error}")
                print("This is normal when loading models for analysis")

    def train(self, steps: int = 0, verbose: int = 1):
        '''
        Train MethylBERT over given steps

        steps: int
            number of steps to train the model
        '''
        return self._iteration(steps, self.train_data, verbose)

    def test(self, test_dataloader: DataLoader):
        '''
        Test/Evaluation of MethylBERT model with given data

        test_dataloader: DataLoader
            Data loader for test data
        '''
        pass

    def load(self, file_path: str):
        '''
        Restore the BERT model store in the given path
        '''
        print(f"Restore the pretrained model from {file_path}")
        self.bert = BertForMaskedLM.from_pretrained(file_path,
            output_attentions=False,
            output_hidden_states=False,
            hidden_dropout_prob=0.1,
            vocab_size = len(self.train_data.dataset.vocab))

        # Initialize the model
        self._setup_model()

    def create_model(self, config_file=None):
        """
        Create a new BERT MLM model from the configuration
        :param config_file: path to the configuration file
        """
        pass

    def _acc(self, pred, label):
        """
        Calculate accuracy between the predicted and the ground-truth values

        :param pred: predicted values
        :param label: ground-truth values
        """

        if type(pred).__module__ != np.__name__:
            pred = pred.numpy()
        if type(label).__module__ != np.__name__:
            label = label.numpy()

        if len(pred.shape) > 1:
            pred = pred.flatten()
        if len(label.shape) > 1:
            label = label.flatten()

        return accuracy_score(y_true=label, y_pred=pred)

    def _multi_class_metrics(self, pred, label, class_names=None):
        """
        Calculate comprehensive multi-class metrics
        
        :param pred: predicted class labels
        :param label: true class labels  
        :param class_names: list of class names for reporting
        :return: dict of metrics
        """
        if type(pred).__module__ != np.__name__:
            pred = pred.numpy()
        if type(label).__module__ != np.__name__:
            label = label.numpy()

        if len(pred.shape) > 1:
            pred = pred.flatten()
        if len(label.shape) > 1:
            label = label.flatten()

        # Basic metrics
        accuracy = accuracy_score(y_true=label, y_pred=pred)
        f1_macro = f1_score(y_true=label, y_pred=pred, average='macro')
        f1_weighted = f1_score(y_true=label, y_pred=pred, average='weighted')
        
        # Per-class accuracy with safe division and fixed label set
        labels = list(range(len(class_names))) if class_names is not None else None
        cm = confusion_matrix(label, pred, labels=labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_accuracy = np.divide(cm.diagonal(), cm.sum(axis=1), where=(cm.sum(axis=1) != 0))
        per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)
        
        # Create detailed classification report
        if class_names is not None:
            # Use labels parameter to include all classes, even if not predicted
            labels = list(range(len(class_names)))
            report = classification_report(label, pred, target_names=class_names, labels=labels, output_dict=True, zero_division=0)
        else:
            report = classification_report(label, pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def plot_training_curves(self, save_path=None):
        """
        Plot training and evaluation curves with reduced noise by limiting to one point per 1000 steps
        """
        def subsample_data(steps, values, min_step_interval=1000):
            """
            Subsample data to have maximum one point per min_step_interval steps
            Always includes the first and last points
            """
            if not steps or len(steps) == 0:
                return [], []
            
            # Convert to numpy arrays for easier manipulation
            steps = np.array(steps)
            values = np.array(values)
            
            # Always include first point
            sampled_indices = [0]
            last_sampled_step = steps[0]
            
            # Sample intermediate points
            for i in range(1, len(steps)):
                if steps[i] - last_sampled_step >= min_step_interval:
                    sampled_indices.append(i)
                    last_sampled_step = steps[i]
            
            # Always include last point (if not already included)
            if sampled_indices[-1] != len(steps) - 1:
                sampled_indices.append(len(steps) - 1)
            
            return steps[sampled_indices].tolist(), values[sampled_indices].tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.train_history['step']:
            train_steps_filtered, train_loss_filtered = subsample_data(
                self.train_history['step'], self.train_history['loss'])
            axes[0, 0].plot(train_steps_filtered, train_loss_filtered, label='Train Loss', color='blue', marker='o', markersize=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.eval_history['step']:
            eval_steps_filtered, eval_loss_filtered = subsample_data(
                self.eval_history['step'], self.eval_history['loss'])
            axes[0, 1].plot(eval_steps_filtered, eval_loss_filtered, label='Validation Loss', color='red', marker='o', markersize=2)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy curves
        if self.train_history['step']:
            train_steps_filtered, train_acc_filtered = subsample_data(
                self.train_history['step'], self.train_history['accuracy'])
            axes[1, 0].plot(train_steps_filtered, train_acc_filtered, label='Train Accuracy', color='green', marker='o', markersize=2)
            axes[1, 0].set_title('Training Accuracy')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.eval_history['step']:
            eval_steps_filtered, eval_acc_filtered = subsample_data(
                self.eval_history['step'], self.eval_history['accuracy'])
            axes[1, 1].plot(eval_steps_filtered, eval_acc_filtered, label='Validation Accuracy', color='orange', marker='o', markersize=2)
            axes[1, 1].set_title('Validation Accuracy')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
            # Print some info about the filtering
            if self.train_history['step']:
                original_points = len(self.train_history['step'])
                filtered_points = len(train_steps_filtered) if 'train_steps_filtered' in locals() else 0
                print(f"Training curve: reduced from {original_points} to {filtered_points} points (1 point per 1000 steps max)")
        else:
            plt.show()
        
        plt.close()

    def plot_confusion_matrix(self, cm, class_names, save_path=None, max_classes=None, **kwargs):
        """
        Plot an improved confusion matrix for multi-class classification.

        Enhancements compared to the old implementation:
        - Shows cell-wise annotations as FULL integer counts only when the
          confusion matrix is smaller than 10x10 (n_classes < 10). For 10x10
          or larger, no per-cell text is drawn to avoid clutter.
        - Uses a perceptually-uniform colormap ("viridis") with optional
          logarithmic scaling so that small values are still distinguishable
          even when errors are spread across many classes
        - Leaves the API unchanged, but supports kwargs: ``log_scale`` and ``cmap``.
        """
        # --- New arguments with backward-compatible defaults -----------------
        log_scale = kwargs.pop("log_scale", True)  # logarithmic color scaling
        cmap      = kwargs.pop("cmap", "viridis")  # perceptually uniform cmap

        n_classes = len(class_names)
        fig_size = max(10, min(0.5 * n_classes, 40))
        plt.figure(figsize=(fig_size, fig_size))

        # ---------------------------------------------------------------------
        # Compute row-normalised matrix (percentages) for coloring
        # ---------------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)

        # Choose colour normalisation
        if log_scale:
            # Avoid zero for LogNorm: set a small floor epsilon
            eps = 1e-6
            positive_vals = cm_normalized[cm_normalized > 0]
            vmin = max(eps, positive_vals.min() if positive_vals.size > 0 else eps)
            norm = mcolors.LogNorm(vmin=vmin, vmax=1.0)
        else:
            norm = None

        # Determine annotation policy: counts only for small matrices
        annotate_small = n_classes < 10
        annot = cm if annotate_small else False
        fmt = 'd' if annotate_small else ''

        sns.heatmap(
            cm_normalized,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            norm=norm,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Row-wise percentage (%)"},
            square=True,
            linewidths=0.5,
            linecolor="white"
        )

        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Cell Type", fontsize=12)
        plt.ylabel("True Cell Type", fontsize=12)

        # Tick label size adjustments
        if n_classes <= 20:
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
        else:
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.yticks(rotation=0, fontsize=7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def print_evaluation_summary(self, metrics, class_names=None):
        """
        Print a comprehensive evaluation summary
        
        :param metrics: dictionary of evaluation metrics
        :param class_names: list of class names
        """
        print("\n" + "="*60)
        print("MULTI-CLASS EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-Class Accuracy (Top 10):")
        per_class_acc = metrics['per_class_accuracy']
        if class_names:
            # Sort by accuracy and show top 10
            sorted_indices = np.argsort(per_class_acc)[::-1]
            for i in range(min(10, len(class_names))):
                idx = sorted_indices[i]
                print(f"  {class_names[idx]:<20}: {per_class_acc[idx]:.4f}")
        else:
            # Show top 10 by index
            sorted_indices = np.argsort(per_class_acc)[::-1]
            for i in range(min(10, len(per_class_acc))):
                idx = sorted_indices[i]
                print(f"  Class {idx:<15}: {per_class_acc[idx]:.4f}")
        
        print(f"\nPer-Class Accuracy (Bottom 10):")
        if class_names:
            for i in range(min(10, len(class_names))):
                idx = sorted_indices[-(i+1)]
                print(f"  {class_names[idx]:<20}: {per_class_acc[idx]:.4f}")
        else:
            for i in range(min(10, len(per_class_acc))):
                idx = sorted_indices[-(i+1)]
                print(f"  Class {idx:<15}: {per_class_acc[idx]:.4f}")
        
        print("="*60)

    def save_best_model(self, file_path="output/bert_best.model"):
        '''
        Save the current model as the best model based on Macro-F1 Score
        '''
        if self.bert is None:
            print("Warning: No model to save. Call create_model() first.")
            return
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        
        if hasattr(self.bert, "read_classifier"):
            classifier_path = os.path.join(os.path.dirname(file_path), "read_classification_model.pickle")
            torch.save(self.bert.read_classifier.state_dict(), classifier_path)
            print(f"Saved read classifier to {classifier_path}")

        if hasattr(self.bert, "dmr_encoder"):
            encoder_path = os.path.join(os.path.dirname(file_path), "dmr_encoder.pickle")
            torch.save(self.bert.dmr_encoder.state_dict(), encoder_path)
            print(f"Saved DMR encoder to {encoder_path}")

        self.bert.to(self.device)
        self.best_model_saved_path = file_path
        print(f"Step:{self.step} Best Model Saved at: {file_path} with Macro-F1 Score: {self.best_model_f1_score:.4f}")

    def save_lowest_loss_model(self, file_path="output/bert.model"):
        '''
        Save the current model as the lowest-loss model
        '''
        if self.bert is None:
            print("Warning: No model to save. Call create_model() first.")
            return
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        
        if hasattr(self.bert, "read_classifier"):
            classifier_path = os.path.join(os.path.dirname(file_path), "read_classification_model.pickle")
            torch.save(self.bert.read_classifier.state_dict(), classifier_path)
            print(f"Saved read classifier to {classifier_path}")
        
        if hasattr(self.bert, "dmr_encoder"):
            encoder_path = os.path.join(os.path.dirname(file_path), "dmr_encoder.pickle")
            torch.save(self.bert.dmr_encoder.state_dict(), encoder_path)
            print(f"Saved DMR encoder to {encoder_path}")
        
        self.bert.to(self.device)
        self.lowest_loss_model_saved_path = file_path
        print(f"Step:{self.step} Lowest-Loss Model Saved at: {file_path} with Loss: {self.best_model_loss:.4f}")
    


    def load_best_model_for_analysis(self):
        '''
        Load the best saved model for final analysis
        '''
        if self.best_model_saved_path and os.path.exists(self.best_model_saved_path):
            print(f"Loading best model from {self.best_model_saved_path} for final analysis...")
            
            # Store current model state temporarily
            current_state = {
                'model_state': self.bert.state_dict(),
                'step': self.step
            }
            
            try:
                # Load the best model using the same logic as the regular load method
                # but treating it as a fine-tuned model
                from methylbert.network import MethylBertEmbeddedDMR
                
                # Get the correct number of DMRs and classes from the dataset
                num_dmrs = self.train_data.dataset.num_dmrs()
                num_classes = self.train_data.dataset.num_classes()  # Get actual number from data
                
                print(f"Loading best model with {num_classes} classes and {num_dmrs} DMR labels")
                print(f"Best model path: {self.best_model_saved_path}")
                print(f"Best model directory contents:")
                best_path_dir = os.path.dirname(self.best_model_saved_path)
                if os.path.exists(best_path_dir):
                    for file in os.listdir(best_path_dir):
                        print(f"  - {file}")
                else:
                    print(f"  Directory {best_path_dir} does not exist!")
                
                # Use the fine-tuned model loading approach since we're loading a saved best model
                print("Calling MethylBertEmbeddedDMR.from_pretrained...")
                try:
                    self.bert = MethylBertEmbeddedDMR.from_pretrained(
                        self.best_model_saved_path,
                        output_attentions=False,
                        output_hidden_states=False,
                        seq_len=self.train_data.dataset.seq_len,
                        loss=self._config.loss,
                        num_labels=num_classes,  # Number of classification classes (HuggingFace convention)
                        num_classes=num_classes,  # Number of classification classes
                        num_dmr_embeddings=num_dmrs,  # DMR embedding size
                        # Add proper label mappings to avoid warnings
                        id2label={i: f"CELL_TYPE_{i}" for i in range(num_classes)},
                        label2id={f"CELL_TYPE_{i}": i for i in range(num_classes)}
                    )
                    print("Model loaded successfully from MethylBertEmbeddedDMR.from_pretrained")
                except Exception as model_load_error:
                    print(f"Failed to load model from MethylBertEmbeddedDMR.from_pretrained: {model_load_error}")
                    raise model_load_error
                
                # Load additional components if they exist
                print("Attempting to load additional components...")
                
                # Try to load DMR encoder
                try:
                    best_path_dir = os.path.dirname(self.best_model_saved_path)
                    dmr_encoder_path = os.path.join(best_path_dir, "dmr_encoder.pickle")
                    if os.path.exists(dmr_encoder_path):
                        print(f"Loading DMR encoder from {dmr_encoder_path}")
                        self.bert.from_pretrained_dmr_encoder(dmr_encoder_path, self.device)
                        print(f"Successfully restored DMR encoder from {dmr_encoder_path}")
                    else:
                        print(f"DMR encoder not found at {dmr_encoder_path}")
                except Exception as e:
                    print(f"Could not load DMR encoder: {e}")
                    print("Continuing without DMR encoder...")
                    
                # Try to load read classifier
                try:
                    best_path_dir = os.path.dirname(self.best_model_saved_path)
                    classifier_path = os.path.join(best_path_dir, "read_classification_model.pickle")
                    if os.path.exists(classifier_path):
                        print(f"Loading read classifier from {classifier_path}")
                        self.bert.from_pretrained_read_classifier(classifier_path, self.device)
                        print(f"Successfully restored read classifier from {classifier_path}")
                    else:
                        print(f"Read classifier not found at {classifier_path}")
                except Exception as e:
                    print(f"Could not load read classifier: {e}")
                    print("Continuing without read classifier...")
                    
                print("Additional component loading completed")
                
                self.bert.to(self.device)
                
                # Ensure model is properly set up on the device
                try:
                    self._setup_model()
                    print(f"Successfully loaded best model from step {self.best_model_step}")
                    return True
                except Exception as setup_error:
                    print(f"Failed to setup best model: {setup_error}")
                    raise setup_error
                
            except Exception as e:
                print(f"Failed to load best model: {e}")
                # Restore current state
                try:
                    self.bert.load_state_dict(current_state['model_state'])
                except Exception as restore_error:
                    print(f"Warning: Could not restore previous model state: {restore_error}")
                return False
        else:
            print("No best model found, using current model for analysis")
            return False

    def load_lowest_loss_model_for_analysis(self):
        '''
        Load the lowest-loss saved model for final analysis
        '''
        if self.lowest_loss_model_saved_path and os.path.exists(self.lowest_loss_model_saved_path):
            print(f"Loading lowest-loss model from {self.lowest_loss_model_saved_path} for final analysis...")
            
            # Store current model state temporarily
            current_state = {
                'model_state': self.bert.state_dict(),
                'step': self.step
            }
            
            try:
                # Load the lowest-loss model using the same logic as the regular load method
                # but treating it as a fine-tuned model
                from methylbert.network import MethylBertEmbeddedDMR
                
                # Get the correct number of DMRs and classes from the dataset
                num_dmrs = self.train_data.dataset.num_dmrs()
                num_classes = self.train_data.dataset.num_classes()  # Get actual number from data
                
                print(f"Loading lowest-loss model with {num_classes} classes and {num_dmrs} DMR labels")
                print(f"Lowest-loss model path: {self.lowest_loss_model_saved_path}")
                print(f"Lowest-loss model directory contents:")
                lowest_loss_path_dir = os.path.dirname(self.lowest_loss_model_saved_path)
                if os.path.exists(lowest_loss_path_dir):
                    for file in os.listdir(lowest_loss_path_dir):
                        print(f"  - {file}")
                else:
                    print(f"  Directory {lowest_loss_path_dir} does not exist!")
                
                # Use the fine-tuned model loading approach since we're loading a saved model
                print("Calling MethylBertEmbeddedDMR.from_pretrained...")
                try:
                    self.bert = MethylBertEmbeddedDMR.from_pretrained(
                        self.lowest_loss_model_saved_path,
                        output_attentions=False,
                        output_hidden_states=False,
                        seq_len=self.train_data.dataset.seq_len,
                        loss=self._config.loss,
                        num_labels=num_classes,  # Number of classification classes (HuggingFace convention)
                        num_classes=num_classes,  # Number of classification classes
                        num_dmr_embeddings=num_dmrs,  # DMR embedding size
                        # Add proper label mappings to avoid warnings
                        id2label={i: f"CELL_TYPE_{i}" for i in range(num_classes)},
                        label2id={f"CELL_TYPE_{i}": i for i in range(num_classes)}
                    )
                    print("Model loaded successfully from MethylBertEmbeddedDMR.from_pretrained")
                except Exception as model_load_error:
                    print(f"Failed to load model from MethylBertEmbeddedDMR.from_pretrained: {model_load_error}")
                    raise model_load_error
                
                # Load additional components if they exist
                print("Attempting to load additional components...")
                
                # Try to load DMR encoder
                try:
                    lowest_loss_path_dir = os.path.dirname(self.lowest_loss_model_saved_path)
                    dmr_encoder_path = os.path.join(lowest_loss_path_dir, "dmr_encoder.pickle")
                    if os.path.exists(dmr_encoder_path):
                        print(f"Loading DMR encoder from {dmr_encoder_path}")
                        self.bert.from_pretrained_dmr_encoder(dmr_encoder_path, self.device)
                        print(f"Successfully restored DMR encoder from {dmr_encoder_path}")
                    else:
                        print(f"DMR encoder not found at {dmr_encoder_path}")
                except Exception as e:
                    print(f"Could not load DMR encoder: {e}")
                    print("Continuing without DMR encoder...")
                
                # Try to load read classifier
                try:
                    lowest_loss_path_dir = os.path.dirname(self.lowest_loss_model_saved_path)
                    classifier_path = os.path.join(lowest_loss_path_dir, "read_classification_model.pickle")
                    if os.path.exists(lowest_loss_path_dir):
                        print(f"Loading read classifier from {classifier_path}")
                        self.bert.from_pretrained_read_classifier(classifier_path, self.device)
                        print(f"Successfully restored read classifier from {classifier_path}")
                    else:
                        print(f"Read classifier not found at {classifier_path}")
                except Exception as e:
                    print(f"Could not load read classifier: {e}")
                    print("Continuing without read classifier...")
                    
                print("Additional component loading completed")
                
                self.bert.to(self.device)
                
                # Ensure model is properly set up on the device
                try:
                    self._setup_model()
                    print(f"Successfully loaded lowest-loss model from step {self.best_loss_step}")
                    return True
                except Exception as setup_error:
                    print(f"Failed to setup lowest-loss model: {setup_error}")
                    raise setup_error
                
            except Exception as e:
                print(f"Failed to load lowest-loss model: {e}")
                # Restore current state
                try:
                    self.bert.load_state_dict(current_state['model_state'])
                except Exception as restore_error:
                    print(f"Warning: Could not restore previous model state: {restore_error}")
                return False
        else:
            print("No lowest-loss model found, using current model for analysis")
            return False

    def _check_f1_score_improvement(self, current_f1_score, verbose=1):
        """
        Check if current Macro-F1 Score is better than previous best and save model if so.
        This method is called during evaluation steps to save the best performing model.
        
        :param current_f1_score: Current Macro-F1 Score from evaluation
        :param verbose: Verbosity level for logging
        """
        if current_f1_score > self.best_model_f1_score:
            # Store the old value for logging
            old_f1_score = self.best_model_f1_score
            
            # Update best F1 score tracking
            self.best_model_f1_score = current_f1_score
            self.best_model_step = self.step
            
            if verbose > 0:
                from tqdm.auto import tqdm as _tqdm
                _tqdm.write(f"Step {self.step}: New best Macro-F1 Score ({current_f1_score:.4f} > {old_f1_score:.4f}). Saving best model...")
            
            # Save the best model
            best_model_path = self.save_path.replace("bert.model", "bert_best.model")
            self.save_best_model(best_model_path)
            
            return True
        return False

    def _check_loss_improvement(self, current_loss, verbose=1):
        """
        Check if current evaluation loss is lower than previous best and save model if so.
        This method is called during evaluation steps to save the lowest-loss model.
        
        :param current_loss: Current evaluation loss
        :param verbose: Verbosity level for logging
        """
        if current_loss < self.best_model_loss:
            # Store the old value for logging
            old_loss = self.best_model_loss
            
            # Update best loss tracking
            self.best_model_loss = current_loss
            self.best_loss_step = self.step
            
            if verbose > 0:
                from tqdm.auto import tqdm as _tqdm
                _tqdm.write(f"Step {self.step}: New best evaluation loss ({current_loss:.4f} < {old_loss:.4f}). Saving lowest-loss model...")
            
            # Save the lowest-loss model to bert.model folder
            self.save_lowest_loss_model(self.save_path)
            
            return True
        return False


class MethylBertPretrainTrainer(MethylBertTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def create_model(self, *args, **kwargs):
        dropout = getattr(self._config, 'dropout', 0.1)
        config = BertConfig(vocab_size = len(self.train_data.dataset.vocab),
                            hidden_dropout_prob=dropout,
                            attention_probs_dropout_prob=dropout,
                            *args, **kwargs)
        self.bert = BertForMaskedLM(config)
        self._setup_model()

    def _eval_iteration(self, data_loader):
        """
        loop over the data_loader for evaluation

        :param data_loader: torch.utils.data.DataLoader for test
        :return: DataFrame,
        """

        predict_res = {"prediction": [], "input": [], "label": [], "mask": []}

        mean_loss = 0
        self.model.eval()

        for i, batch in enumerate(data_loader):

            data = {key: value.to(self.device) for key, value in batch.items()}

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu",
                                    enabled=self._config.amp):
                        mask_lm_output = self.model.forward(input_ids = data["input"],
                                                        masked_lm_labels = data["label"])

                mean_loss += mask_lm_output[0].mean().item()/len(data_loader)
                predict_res["prediction"].append(np.argmax(mask_lm_output[1].cpu().detach(), axis=-1))
                predict_res["input"].append(data["input"].cpu().detach())
                predict_res["label"].append(data["label"].cpu().detach())
                predict_res["mask"].append(data["mask"].cpu().detach())

            if self._config.eval:
                print("Batch %d/%d is done...."%(i, len(data_loader)))

            del mask_lm_output
            del data

        # Integrate all results
        predict_res["prediction"] = np.concatenate(predict_res["prediction"], axis=0)
        predict_res["input"] = np.concatenate(predict_res["input"], axis=0)
        predict_res["label"] = np.concatenate(predict_res["label"],  axis=0)
        predict_res["mask"] = np.concatenate(predict_res["mask"],  axis=0)

        self.model.train()
        return predict_res, np.mean(mean_loss)


    def _iteration(self, steps, data_loader, verbose):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param steps: total steps to train
        :param data_loader: torch.utils.data.DataLoader for training
        :param warm_up: number of steps for warming up the learning rate
        :return: None
        """
        predict_res = {"prediction": [], "input": [], "label": []}
        self.step = 0

        if os.path.exists(self.f_train):
            os.remove(self.f_train)

        with open(self.f_train, "a") as f_perform:
            f_perform.write("step\tloss\tacc\tlr\n")

        if os.path.exists(self.f_eval):
            os.remove(self.f_eval)

        with open(self.f_eval, "a") as f_perform:
            f_perform.write("step\ttest_acc\ttest_loss\n")


        # Set up a learning rate scheduler
        self.scheduler = learning_rate_scheduler(self.optim,
                                                 num_warmup_steps=self._config.warmup_step,
                                                 num_training_steps=steps,
                                                 decrease_steps=self._config.decrease_steps)

        # Set up configuration for train iteration
        global_step_loss = 0
        local_step = 0

        epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1
        self.model.zero_grad()
        self.model.train()
        train_prediction_res = {"prediction":[], "label":[]}

        scaler = GradScaler("cuda" if self._config.with_cuda else "cpu") if self._config.amp else None

        duration = 0
        for epoch in range(epochs):
            for i, batch in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in batch.items()}

                start = time.time()

                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu",
                                    enabled=self._config.amp):
                    mask_lm_output = self.model.forward(input_ids = data["bert_input"],
                                                masked_lm_labels = data["bert_label"]) 

                loss = mask_lm_output[0]

                # Concatenate predicted sequences for the evaluation
                train_prediction_res["prediction"].append(np.argmax(mask_lm_output[1].cpu().detach(), axis=-1))
                train_prediction_res["label"].append(data["bert_label"].cpu().detach())

                # Calculate loss and back-propagation
                if "cuda" in self.device.type:
                    loss = loss.mean()
                loss = loss/self._config.gradient_accumulation_steps
                scaler.scale(loss).backward() if self._config.amp else loss.backward()

                global_step_loss += loss.item()
                duration += time.time() - start

                # Gradient accumulation
                if (local_step+1) % self._config.gradient_accumulation_steps == 0:

                    if self._config.amp:
                        scaler.unscale_(self.optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        self.optim.step()

                    self.scheduler.step()
                    self.model.zero_grad()

                    # Evaluation with both train and testdata
                    if self.test_data is not None and self.step % self._config.eval_freq == 0:

                        test_pred, test_loss = self._eval_iteration(self.test_data)
                        idces = np.where(test_pred["label"]>=0)
                        test_pred_acc = self._acc(test_pred["prediction"][idces[0], idces[1]],
                                                  test_pred["label"][idces[0], idces[1]])

                        with open(self.f_eval, "a") as f_perform:
                            f_perform.write("\t".join([str(self.step), str(test_pred_acc), str(test_loss)]) +"\n")

                        del test_pred

                    if self.step % self._config.log_freq == 0:
                        from tqdm.auto import tqdm as _tqdm
                        _tqdm.write(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
                        _tqdm.write(f"Running time for iter = {duration:.2f}s")
                        sys.stdout.flush()

                    if self.min_loss > global_step_loss:
                        print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s"%(self.step, global_step_loss, self.min_loss, self.save_path))
                        self.save(self.save_path)
                        self.min_loss = global_step_loss

                    # Save the step info (step, loss, lr, acc)
                    with open(self.f_train, "a") as f_perform:

                        train_prediction_res["prediction"] = np.concatenate(train_prediction_res["prediction"], axis=0)
                        train_prediction_res["label"] = np.concatenate(train_prediction_res["label"],  axis=0)

                        idces = np.where(train_prediction_res["label"]>=0)
                        train_pred_acc = self._acc(train_prediction_res["prediction"][idces[0], idces[1]],
                            train_prediction_res["label"][idces[0], idces[1]])

                        f_perform.write("\t".join([str(self.step), str(global_step_loss), str(train_pred_acc), str(self.optim.param_groups[0]["lr"])])+"\n")

                    self.step += 1

                    duration=0
                    global_step_loss = 0
                    del train_prediction_res
                    train_prediction_res = {"prediction":[], "label":[]}

                if steps == self.step:
                    break
                local_step+=1

            if steps == self.step:
                break


class MethylBertFinetuneTrainer(MethylBertTrainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def summary(self):
		'''
		Print the summary of the MethylBERT model
		'''

		print(self.model)

	def _compute_class_weights(self) -> torch.Tensor:
		"""Compute inverse-frequency class weights from the training dataset.
		Normalizes weights to have mean 1 for stable loss scaling.
		"""
		if not hasattr(self.train_data.dataset, 'ctype_label_count'):
			return None
		counts = np.array(self.train_data.dataset.ctype_label_count, dtype=np.float64)
		eps = 1e-8
		inv_freq = 1.0 / (counts + eps)
		# Normalize to mean 1
		weights = inv_freq * (counts.size / inv_freq.sum())
		return torch.tensor(weights, dtype=torch.float32)

	def _compute_focal_weight(self):
		"""Compute per-class focal weights as normalized inverse label frequency.
		This replaces the old focal_alpha computation for the new softmax-based focal loss.
		"""
		if not hasattr(self.train_data.dataset, 'ctype_label_count'):
			return None
		counts = np.array(self.train_data.dataset.ctype_label_count, dtype=np.float64)
		eps = 1e-8
		inv_freq = 1.0 / (counts + eps)
		# Normalize to mean 1 for stable loss scaling (same as class weights)
		weights = inv_freq * (counts.size / inv_freq.sum())
		weights = weights.astype(np.float32)
		return weights.tolist()

	def _compute_decoupled_alpha_pos(self):
		"""Compute per-class positive alpha for decoupled focal loss.
		alpha_pos = (1 - freq).clamp(0.6, 0.9) where freq is normalized frequency.
		"""
		import torch
		if not hasattr(self.train_data.dataset, 'ctype_label_count'):
			return None
		counts = np.array(self.train_data.dataset.ctype_label_count, dtype=np.float64)
		total_count = counts.sum()
		freq = counts / total_count  # normalized frequency
		freq_tensor = torch.tensor(freq, dtype=torch.float32)
		alpha_pos = (1 - freq_tensor).clamp(0.6, 0.9)  # strong pos weight
		return alpha_pos.tolist()

	def _compute_decoupled_alpha_neg(self):
		"""Compute per-class negative alpha for decoupled focal loss.
		alpha_neg = torch.ones_like(alpha_pos) - uniform weight for negatives.
		"""
		import torch
		if not hasattr(self.train_data.dataset, 'ctype_label_count'):
			return None
		counts = np.array(self.train_data.dataset.ctype_label_count, dtype=np.float64)
		num_classes = len(counts)
		alpha_neg = torch.ones(num_classes, dtype=torch.float32)
		return alpha_neg.tolist()

	def _analyze_evaluation_results(self, eval_pred, eval_loss, class_names, eval_logits):
		"""
		Analyze and print evaluation results with per-class breakdown.
		
		Args:
			eval_pred: Evaluation predictions dictionary
			eval_loss: Evaluation loss value
			class_names: Dictionary mapping class indices to names
			eval_logits: Model output logits for confidence calculation
		"""
		with torch.no_grad():
			pred_ctype_labels = eval_pred["pred_ctype_label"]
			true_ctype_labels = eval_pred["ctype_label"]
			
			# Convert to tensors if they aren't already
			if not isinstance(pred_ctype_labels, torch.Tensor):
				pred_ctype_labels = torch.tensor(pred_ctype_labels)
			if not isinstance(true_ctype_labels, torch.Tensor):
				true_ctype_labels = torch.tensor(true_ctype_labels)
			
			# Convert eval_logits to tensor if it's a numpy array
			if isinstance(eval_logits, np.ndarray):
				eval_logits = torch.tensor(eval_logits)
			
			# Compute per-sample accuracy
			correct_predictions = (pred_ctype_labels == true_ctype_labels).float()
			
			# Get unique classes in the evaluation data
			unique_classes = torch.unique(true_ctype_labels)
			num_classes = len(unique_classes)
			
			# Analyze by class
			class_contributions = {}
			for class_idx in unique_classes:
				class_idx = class_idx.item()
				class_mask = (true_ctype_labels == class_idx)
				if class_mask.sum() > 0:
					class_correct = correct_predictions[class_mask].mean().item()
					
					# Calculate confidence for this class (probability assigned to true class)
					class_logits = eval_logits[class_mask]
					class_true_labels = true_ctype_labels[class_mask]
					
					# Get probabilities for the true classes
					if hasattr(self.bert, 'loss') and self.bert.loss == 'focal':
						# For focal loss, use sigmoid
						class_probs = torch.sigmoid(class_logits)
					else:
						# For cross-entropy, use softmax
						class_probs = torch.softmax(class_logits, dim=1)
					
					# Get confidence for true class
					true_class_probs = class_probs[torch.arange(len(class_true_labels)), class_true_labels]
					class_conf = true_class_probs.mean().item()
					
					class_contributions[class_idx] = {
						'accuracy': class_correct,
						'confidence': class_conf
					}
			
			# Calculate overall accuracy and confidence
			overall_acc = correct_predictions.mean().item()
			
			# Calculate overall confidence (average confidence across all samples)
			if hasattr(self.bert, 'loss') and self.bert.loss == 'focal':
				all_probs = torch.sigmoid(eval_logits)
			else:
				all_probs = torch.softmax(eval_logits, dim=1)
			
			all_true_class_probs = all_probs[torch.arange(len(true_ctype_labels)), true_ctype_labels]
			overall_conf = all_true_class_probs.mean().item()
			
			# Print evaluation analysis
			print(f"  EVALUATION Results Analysis (Step {self.step}):")
			print(f"    Overall - Loss: {eval_loss:.4f}, Accuracy: {overall_acc:.3f}, Avg Confidence: {overall_conf:.3f}")
			print(f"    Per-class breakdown:")
			
			for class_idx, stats in class_contributions.items():
				class_name = class_names.get(class_idx, f"Class_{class_idx}") if class_idx in class_names else f"Class_{class_idx}"
				print(f"      {class_name:<15}: Acc={stats['accuracy']:.3f}, Conf={stats['confidence']:.3f}")

	def _analyze_loss_contribution(self, logits, true_labels, loss_value):
		"""
		Analyze and print loss contribution by class for sanity checking.
		
		Args:
			logits: Model output logits [batch_size, num_classes]
			true_labels: True class labels [batch_size]
			loss_value: Computed loss value (scalar)
		"""
		with torch.no_grad():
			# Convert logits to tensor if it's a numpy array
			if isinstance(logits, np.ndarray):
				logits = torch.tensor(logits)
			
			batch_size = logits.shape[0]
			num_classes = logits.shape[1]
			
			# Get predicted classes
			predicted_classes = torch.argmax(logits, dim=1)
			
			# Compute probabilities
			if hasattr(self.bert, 'loss') and self.bert.loss == 'focal':
				# For focal loss, use sigmoid
				probs = torch.sigmoid(logits)
			else:
				# For cross-entropy, use softmax
				probs = torch.softmax(logits, dim=1)
			
			# Compute per-sample contributions
			correct_predictions = (predicted_classes == true_labels).float()
			incorrect_predictions = 1.0 - correct_predictions
			
			# Get confidence for true class
			true_class_probs = probs[torch.arange(batch_size), true_labels]
			
			# Analyze by class
			class_contributions = {}
			for class_idx in range(num_classes):
				class_mask = (true_labels == class_idx)
				if class_mask.sum() > 0:
					class_correct = correct_predictions[class_mask].mean().item()
					class_conf = true_class_probs[class_mask].mean().item()
					class_count = class_mask.sum().item()
					
					class_contributions[class_idx] = {
						'accuracy': class_correct,
						'confidence': class_conf,
						'count': class_count
					}
			
			# Always return the analysis results, let caller decide when to print
			overall_acc = correct_predictions.mean().item()
			overall_conf = true_class_probs.mean().item()
			
			# Get class names if available
			class_names = None
			if hasattr(self.train_data.dataset, 'int_to_ctype'):
				class_names = self.train_data.dataset.int_to_ctype
			
			print(f"  Loss Contribution Analysis (Step {self.step}):")
			print(f"    Overall - Loss: {loss_value:.4f}, Accuracy: {overall_acc:.3f}, Avg Confidence: {overall_conf:.3f}")
			print(f"    Per-class breakdown:")
			
			for class_idx, stats in class_contributions.items():
				class_name = class_names.get(class_idx, f"Class_{class_idx}") if class_names else f"Class_{class_idx}"
				print(f"      {class_name:<15}: Acc={stats['accuracy']:.3f}, Conf={stats['confidence']:.3f}, Count={stats['count']:3d}")

	# --- NEW: Initialize classifier bias from validation priors to counter argmax drift ---
	def _compute_validation_priors(self):
		"""Compute class priors from the validation/test dataset.
		Returns a numpy array of shape [num_classes] or None if unavailable.
		"""
		if self.test_data is None or not hasattr(self.test_data, 'dataset'):
			return None
		dataset = self.test_data.dataset
		if not hasattr(dataset, 'ctype_label_count'):
			return None
		counts = np.array(dataset.ctype_label_count, dtype=np.float64)
		if counts.sum() <= 0:
			return None
		priors = counts / counts.sum()
		return priors

	def _init_classifier_bias_from_validation(self):
		"""Set the final classifier bias using validation class frequencies.
		- For multi-class softmax (cross_entropy): b_j = log(pi_j)
		- For sigmoid-based losses (focal/bce/decoupled_focal): b_j = logit(pi_j)
		Silently no-ops if prerequisites are missing.
		"""
		try:
			priors = self._compute_validation_priors()
			if priors is None:
				return
			# Determine target layer and out_features
			layer = None
			if hasattr(self.bert, 'read_classifier') and len(self.bert.read_classifier) > 0:
				layer = self.bert.read_classifier[-1]
			if layer is None or not hasattr(layer, 'bias'):
				return
			out_features = layer.out_features
			# Align priors length to classifier out_features
			if len(priors) != out_features:
				aligned = np.full(out_features, 1.0 / max(out_features, 1), dtype=np.float64)
				for i in range(min(out_features, len(priors))):
					aligned[i] = priors[i]
				priors = aligned / aligned.sum()
			import torch
			p = torch.tensor(priors, dtype=torch.float32, device=self.device).clamp(1e-4, 1-1e-4)
			with torch.no_grad():
				if getattr(self._config, 'loss', 'cross_entropy') == 'cross_entropy':
					bias = torch.log(p)
				else:
					bias = torch.log(p / (1 - p))
				# Set on self.bert
				layer.bias.copy_(bias.to(layer.bias.device))
				# If model is DataParallel, also set there
				if hasattr(self, 'model'):
					m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
					if hasattr(m, 'read_classifier') and len(m.read_classifier) > 0 and hasattr(m.read_classifier[-1], 'bias'):
						m.read_classifier[-1].bias.copy_(bias.to(m.read_classifier[-1].bias.device))
		except Exception as e:
			# Do not interrupt training for non-critical init
			print(f"Warning: could not initialize classifier bias from validation priors: {e}")

	def create_model(self, config_file: str = None):
		'''
		Create a new MethylBERT model from the configuration
		'''

		dropout = getattr(self._config, 'dropout', 0.1)
		# Get actual number of classes from dataset
		actual_num_classes = self.train_data.dataset.num_classes()
		print(f"Creating model with {actual_num_classes} classes (from dataset)")
		
		config = MethylBERTConfig.from_pretrained(config_file,
			output_attentions=False,
			output_hidden_states=False,
			hidden_dropout_prob=dropout,
			attention_probs_dropout_prob=dropout,
			vocab_size = len(self.train_data.dataset.vocab),
			loss=self._config.loss,
			num_labels=actual_num_classes,  # Number of classification classes (HuggingFace convention)
			num_classes=actual_num_classes,  # Number of classification classes
			num_dmr_embeddings=self.train_data.dataset.num_dmrs(),  # DMR embedding size
			# Add proper label mappings
			id2label={i: f"CELL_TYPE_{i}" for i in range(actual_num_classes)},
			label2id={f"CELL_TYPE_{i}": i for i in range(actual_num_classes)})

		# Label smoothing for cross-entropy
		setattr(config, 'label_smoothing', getattr(self._config, 'label_smoothing', 0.0))
		# Focal loss hyperparameters
		# focal_alpha parameter is deprecated - now using focal_weight
		setattr(config, 'focal_gamma', getattr(self._config, 'focal_gamma', 2.0))

		# Configure focal weight (auto-compute if not provided as a valid per-class vector)
		if config.loss == 'focal':
			weight_cfg = getattr(self._config, 'focal_weight', None)
			n_classes = getattr(self._config, 'num_classes', None)
			if n_classes is None and hasattr(self.train_data.dataset, 'num_classes'):
				n_classes = self.train_data.dataset.num_classes()
			weight_to_use = None
			if isinstance(weight_cfg, (list, tuple)):
				if n_classes is not None and len(weight_cfg) == n_classes:
					weight_to_use = [float(x) for x in weight_cfg]
					print(f"Using config-provided per-class focal weight (len={len(weight_to_use)}). First 10: {weight_to_use[:10]}")
				else:
					print(f"Warning: focal_weight length {len(weight_cfg)} does not match num classes {n_classes}. Recomputing automatically from label frequencies.")
					weight_to_use = self._compute_focal_weight()
			elif weight_cfg is not None:
				print(f"Warning: focal_weight must be a list/tuple for per-class weights or None for auto-computation. Got {type(weight_cfg)}. Auto-computing.")
				weight_to_use = self._compute_focal_weight()
			else:
				weight_to_use = self._compute_focal_weight()
				if weight_to_use is not None:
					print(f"Using auto-computed per-class focal weight (len={len(weight_to_use)}). First 10: {weight_to_use[:10]}")
					# Example (from dataset prints):
					#   # of reads in each label:  [172279. 169940. 118181. 148517. 143046.]
					#   Total number of unique cell types: 5
					#   Cell type mapping (first 10): {'Blood-B': 0, 'Blood-Granul': 1, 'Blood-Mono+Macro': 2, 'Blood-NK': 3, 'Blood-T': 4}
					#   -> Computed focal weight (normalized inverse frequency): [.. per-class values ..]
			setattr(config, 'focal_weight', weight_to_use)

		# Configure decoupled focal loss parameters
		elif config.loss == 'decoupled_focal':
			n_classes = getattr(self._config, 'num_classes', None)
			if n_classes is None and hasattr(self.train_data.dataset, 'num_classes'):
				n_classes = self.train_data.dataset.num_classes()
			
			alpha_pos_cfg = getattr(self._config, 'focal_alpha_pos', None)
			alpha_neg_cfg = getattr(self._config,
			'focal_alpha_neg', None)
			
			alpha_pos_to_use = None
			alpha_neg_to_use = None
			
			# Handle alpha_pos
			if isinstance(alpha_pos_cfg, (list, tuple)):
				if n_classes is not None and len(alpha_pos_cfg) == n_classes:
					alpha_pos_to_use = [float(x) for x in alpha_pos_cfg]
					print(f"Using config-provided per-class focal alpha_pos (len={len(alpha_pos_to_use)}). First 10: {alpha_pos_to_use[:10]}")
				else:
					print(f"Warning: focal_alpha_pos length {len(alpha_pos_cfg)} does not match num classes {n_classes}. Computing automatically.")
					alpha_pos_to_use = self._compute_decoupled_alpha_pos()
			elif isinstance(alpha_pos_cfg, (int, float)):
				alpha_pos_to_use = float(alpha_pos_cfg)
				print(f"Using config-provided scalar focal alpha_pos: {alpha_pos_to_use}")
			else:
				alpha_pos_to_use = self._compute_decoupled_alpha_pos()
				if alpha_pos_to_use is not None:
					print(f"Using auto-computed per-class focal alpha_pos (len={len(alpha_pos_to_use)}). First 10: {alpha_pos_to_use[:10]}")
			
			# Handle alpha_neg  
			if isinstance(alpha_neg_cfg, (list, tuple)):
				if n_classes is not None and len(alpha_neg_cfg) == n_classes:
					alpha_neg_to_use = [float(x) for x in alpha_neg_cfg]
					print(f"Using config-provided per-class focal alpha_neg (len={len(alpha_neg_to_use)}). First 10: {alpha_neg_to_use[:10]}")
				else:
					print(f"Warning: focal_alpha_neg length {len(alpha_neg_cfg)} does not match num classes {n_classes}. Computing automatically.")
					alpha_neg_to_use = self._compute_decoupled_alpha_neg()
			elif isinstance(alpha_neg_cfg, (int, float)):
				alpha_neg_to_use = float(alpha_neg_cfg)
				print(f"Using config-provided scalar focal alpha_neg: {alpha_neg_to_use}")
			else:
				alpha_neg_to_use = self._compute_decoupled_alpha_neg()
				if alpha_neg_to_use is not None:
					print(f"Using auto-computed per-class focal alpha_neg (len={len(alpha_neg_to_use)}). First 10: {alpha_neg_to_use[:10]}")
			
			setattr(config, 'focal_alpha_pos', alpha_pos_to_use)
			setattr(config, 'focal_alpha_neg', alpha_neg_to_use)

		class_weights = None
		if config.loss == 'cross_entropy':
			class_weights = self._compute_class_weights()
			if class_weights is not None:
				print("Using class-weighted cross entropy. Weights (first 10):", class_weights[:10])

		self.bert = MethylBertEmbeddedDMR(config=config,
										  seq_len=self.train_data.dataset.seq_len,
										  num_classes=actual_num_classes,
										  class_weights=class_weights)

		# Initialize the BERT Language Model, with BERT model
		self._setup_model()
		# Initialize classifier bias from validation priors
		self._init_classifier_bias_from_validation()

	def _eval_iteration(self, data_loader: DataLoader, return_logits: bool = False):
		"""
		loop over the data_loader for eval/test

		:param data_loader: torch.utils.data.DataLoader for test
		:return: DataFrame,
		"""
		
		predict_res = {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
		logits = list()
		
		mean_loss = 0
		self.model.eval()
		with torch.no_grad():
			for i, batch in enumerate(data_loader):
				# 0. batch_data will be sent into the device(GPU or cpu)
				data = {key: value.to(self.device) for key, value in batch.items() if type(value) != list}
				
				with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", enabled=self._config.amp):
					attention_mask = (data["dna_seq"] != self.train_data.dataset.vocab.pad_index).long()
					mask_lm_output = self.model.forward(step=self.step,
														input_ids = data["dna_seq"],
														attention_mask=attention_mask,
														token_type_ids=data["methyl_seq"],
														labels = data["dmr_label"],
														ctype_label=data["ctype_label"])

				loss = mask_lm_output["loss"].mean().item() if "cuda" in self.device.type else mask_lm_output["loss"].item()
				mean_loss += loss/len(data_loader)
				
				if self._config.with_cuda and torch.cuda.device_count() > 1:
					torch.cuda.synchronize()
				
				predict_res["dmr_label"].append(data["dmr_label"].detach().cpu())
				predict_res["pred_ctype_label"].append(torch.argmax(mask_lm_output["classification_logits"], dim=-1).detach().cpu())
				predict_res["ctype_label"].append(data["ctype_label"].detach().cpu())
				
				if return_logits:
					logits.append(mask_lm_output["classification_logits"].detach().cpu().numpy())
				
				del mask_lm_output
				del data

		predict_res["dmr_label"] = np.concatenate(predict_res["dmr_label"],  axis=0)
		predict_res["ctype_label"] = np.concatenate(predict_res["ctype_label"],  axis=0)
		predict_res["pred_ctype_label"] = np.concatenate(predict_res["pred_ctype_label"], axis=0)
		
		self.model.train()
		
		if not return_logits:
			return predict_res, mean_loss
		else:
			return predict_res, mean_loss, np.concatenate(logits, axis=0)

	def _iteration(self, steps, data_loader, verbose = 1):
		"""
		loop over the data_loader for training or testing
		if on train status, backward operation is activated
		and also auto save the model every peoch

		:param steps: total steps to train
		:param data_loader: torch.utils.data.DataLoader for training
		:param warm_up: number of steps for warming up the learning rate
		:return: None
		"""
		self.step = 0

		if os.path.exists(self.f_train):
			os.remove(self.f_train)

		with open(self.f_train, "w") as f_perform:
			f_perform.write("step\tloss\tctype_acc\tlr\n")

		if os.path.exists(self.f_eval):
			os.remove(self.f_eval)

		with open(self.f_eval, "w") as f_perform:
			f_perform.write("step\tloss\tctype_acc\n")

		# Set up a learning rate scheduler
		self.scheduler = learning_rate_scheduler(self.optim,
														 num_warmup_steps=self._config.warmup_step,
														 num_training_steps=steps,
														 decrease_steps=self._config.decrease_steps)
		global_step_loss = 0
		local_step = 0

		epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1

		self.model.zero_grad()
		if verbose > 0:
			print(self.model.training)
		self.model.train()
		train_prediction_res = {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}

		device_type = "cuda" if self._config.with_cuda else "cpu"
		scaler = GradScaler(device_type) if self._config.amp else None

		# Track latest evaluation metrics and class names safely across steps
		last_eval_loss = None
		last_eval_acc = None
		class_names_global = list(self.train_data.dataset.int_to_ctype.values()) if hasattr(self.train_data.dataset, 'int_to_ctype') else None

		# Suppress transformers warnings
		import warnings
		warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

		duration = 0
		# Create a single progress bar for all steps
		progress_bar = tqdm(total=steps, desc="Training Steps")
		for epoch in range(epochs):
			for i, batch in enumerate(data_loader):
				# 0. batch_data will be sent into the device(GPU or cpu)
				data = {key: value.to(self.device) for key, value in batch.items() if type(value) != list}

				start = time.time()
				with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu",
									enabled=self._config.amp):
					attention_mask = (data["dna_seq"] != self.train_data.dataset.vocab.pad_index).long()
					mask_lm_output = self.model.forward(step=self.step,
											input_ids=data["dna_seq"],
											attention_mask=attention_mask,
											token_type_ids=data["methyl_seq"],
											labels=data["dmr_label"],
											ctype_label=data["ctype_label"])
				loss = mask_lm_output["loss"]

				# Concatenate predicted sequences for the evaluation
				train_prediction_res["dmr_label"].append(data["dmr_label"].detach().cpu())

				# Cell-type classification
				train_prediction_res["pred_ctype_label"].append(np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1))
				train_prediction_res["ctype_label"].append(data["ctype_label"].detach().cpu())

				# Calculate loss and back-propagation
				loss = mask_lm_output["loss"].mean() if "cuda" in self.device.type else mask_lm_output["loss"]
				loss = loss/self._config.gradient_accumulation_steps
				
				# Only analyze loss contribution at log frequency
				if self.step % self._config.log_freq == 0:
					self._analyze_loss_contribution(
						mask_lm_output["classification_logits"],
						data["ctype_label"],
						loss.item()
					)
				
				scaler.scale(loss).backward() if self._config.amp else loss.backward()

				loss_val = loss.item()
				global_step_loss += loss_val

				duration += time.time() - start
				# Gradient accumulation
				if (local_step+1) % self._config.gradient_accumulation_steps == 0:
					gradient_accum_start = time.time()
					if self._config.amp:
						scaler.unscale_(self.optim)
						nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
						scaler.step(self.optim)
						scaler.update()
					else:
						nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
						self.optim.step()

					self.scheduler.step()
					self.model.zero_grad()
					


				if local_step % self._config.eval_freq == 0 and local_step > 0:
					# Evaluation
					eval_pred, eval_loss, eval_logits = self._eval_iteration(self.test_data, return_logits=True)
					
					# Calculate multi-class metrics
					class_names = list(self.train_data.dataset.int_to_ctype.values()) if hasattr(self.train_data.dataset, 'int_to_ctype') else class_names_global
					eval_metrics = self._multi_class_metrics(eval_pred["pred_ctype_label"], eval_pred["ctype_label"], class_names)
					eval_acc = eval_metrics['accuracy']
					eval_f1_macro = eval_metrics['f1_macro']  # Get Macro-F1 Score
					last_eval_loss = float(eval_loss)
					last_eval_acc = float(eval_acc)

					# Check if this is the best model based on Macro-F1 Score
					self._check_f1_score_improvement(eval_f1_macro, verbose)
					
					# Check if this is the best model based on evaluation loss
					self._check_loss_improvement(eval_loss, verbose)

					# Store evaluation history
					self.eval_history['step'].append(self.step)
					self.eval_history['loss'].append(eval_loss)
					self.eval_history['accuracy'].append(eval_acc)

					# Print detailed evaluation analysis with per-class breakdown
					self._analyze_evaluation_results(eval_pred, eval_loss, class_names, eval_logits)

					with open(self.f_eval, "a") as f_perform:
						f_perform.write("\t".join([str(self.step), str(eval_loss), str(eval_acc)]) +"\n")

					del eval_pred

					if self.step % self._config.log_freq == 0:
						if verbose > 0:
							# Calculate time estimation
							current_time = time.time()
							if not hasattr(self, 'training_start_time'):
								self.training_start_time = current_time
							
							elapsed_time = current_time - self.training_start_time
							steps_completed = self.step + 1
							steps_remaining = steps - steps_completed
							
							if steps_completed > 0:
								avg_time_per_step = elapsed_time / steps_completed
								estimated_time_remaining = avg_time_per_step * steps_remaining
								
								# Convert to human readable format
								def format_time(seconds):
									hours = int(seconds // 3600)
									minutes = int((seconds % 3600) // 60)
									secs = int(seconds % 60)
									if hours > 0:
										return f"{hours}h {minutes}m {secs}s"
									elif minutes > 0:
										return f"{minutes}m {secs}s"
									else:
										return f"{secs}s"
								
								eta_str = format_time(estimated_time_remaining)
								elapsed_str = format_time(elapsed_time)
								
								tqdm.write(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
								tqdm.write(f"Running time for iter = {duration:.2f}s")
								tqdm.write(f"Total elapsed: {elapsed_str} | ETA: {eta_str}")
								tqdm.write(f"Validation - Loss: {last_eval_loss:.4f}, Accuracy: {last_eval_acc:.4f}")
								sys.stdout.flush()
							else:
								tqdm.write(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
								tqdm.write(f"Running time for iter = {duration:.2f}s")
								tqdm.write(f"Validation - Loss: {last_eval_loss:.4f}, Accuracy: {last_eval_acc:.4f}")
								sys.stdout.flush()

				# Save the step info (step, loss, lr, acc)
				with open(self.f_train, "a") as f_perform:

					train_prediction_res["dmr_label"] = np.concatenate(train_prediction_res["dmr_label"],  axis=0)
					train_prediction_res["pred_ctype_label"] = np.concatenate(train_prediction_res["pred_ctype_label"], axis=0)
					train_prediction_res["ctype_label"] = np.concatenate(train_prediction_res["ctype_label"],  axis=0)
					
					# Calculate training metrics (use global class names if local not set)
					train_metrics = self._multi_class_metrics(train_prediction_res["pred_ctype_label"], train_prediction_res["ctype_label"], locals().get('class_names', class_names_global))
					train_ctype_acc = train_metrics['accuracy']

					# Store training history
					self.train_history['step'].append(self.step)
					self.train_history['loss'].append(global_step_loss)
					self.train_history['accuracy'].append(train_ctype_acc)
					self.train_history['lr'].append(self.optim.param_groups[0]["lr"])

					f_perform.write("\t".join([str(self.step), str(global_step_loss), str(train_ctype_acc),  str(self.optim.param_groups[0]["lr"])])+"\n")

				progress_bar.set_postfix(
					step=self.step,
					loss=f"{global_step_loss:.4f}",
					eval_loss=f"{(last_eval_loss if last_eval_loss is not None else float('nan')):.4f}",
					eval_acc=f"{(last_eval_acc if last_eval_acc is not None else float('nan')):.3f}",
					lr=f"{self.optim.param_groups[0]['lr']:.6f}"
				)
				# Reset prediction result
				del train_prediction_res
				train_prediction_res =  {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}

				# --- Ensure logging happens at the specified frequency even when no evaluation is triggered ---
				# Avoid duplicate prints on evaluation steps
				if (self.step % self._config.log_freq == 0) and (local_step % self._config.eval_freq != 0 or local_step == 0):
					if verbose > 0:
						tqdm.write(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
						tqdm.write(f"Running time for iter = {duration:.2f}s")
						sys.stdout.flush()
				# ------------------------------------------------------------------------------

				self.step += 1
				duration=0
				global_step_loss = 0

				progress_bar.update(1)

				if steps == self.step:
					break
				local_step+=1

			if steps == self.step:
				break

		progress_bar.close()

		# Final evaluation and plotting
		if verbose > 0:
			print("\n" + "="*60)
			print("TRAINING COMPLETED - GENERATING FINAL EVALUATION")
			print("="*60)
			
			# Generate plots directory
			plots_dir = os.path.join(self.save_path, "plots")
			if not os.path.exists(plots_dir):
				os.mkdir(plots_dir)
			
			# Plot training curves (only need to do this once)
			self.plot_training_curves(save_path=os.path.join(plots_dir, "training_curves.png"))
			
			# Get class names once
			class_names = list(self.train_data.dataset.int_to_ctype.values()) if hasattr(self.train_data.dataset, 'int_to_ctype') else None
			
			# 1. Evaluate with BEST F1 MODEL (bert_best.model)
			print("\n" + "="*40)
			print("EVALUATING BEST F1 MODEL (bert_best.model)")
			print("="*40)
			
			best_f1_model_loaded = self.load_best_model_for_analysis()
			if best_f1_model_loaded:
				print(f"Using best F1 model from step {self.best_model_step} (F1: {self.best_model_f1_score:.4f}) for analysis")
				
				# Final evaluation with logits for low-confidence analysis
				best_eval_pred, best_eval_loss, best_eval_logits = self._eval_iteration(self.test_data, return_logits=True)
				best_metrics = self._multi_class_metrics(best_eval_pred["pred_ctype_label"], best_eval_pred["ctype_label"], class_names)
				
				# Print comprehensive evaluation summary
				print(f"\nBEST F1 MODEL EVALUATION (Step {self.best_model_step}):")
				self.print_evaluation_summary(best_metrics, class_names)
				
				# Plot confusion matrix for best F1 model
				self.plot_confusion_matrix(
					best_metrics['confusion_matrix'], 
					class_names, 
					save_path=os.path.join(plots_dir, "confusion_matrix_best_f1.png"),
					max_classes=20  # Show top 20 most frequent classes for compactness
				)
				print(f"Best F1 model confusion matrix saved to: {os.path.join(plots_dir, 'confusion_matrix_best_f1.png')}")
			else:
				print("No best F1 model found, skipping best F1 model evaluation")
			
			# 2. Evaluate with LOWEST LOSS MODEL (bert.model)
			print("\n" + "="*40)
			print("EVALUATING LOWEST LOSS MODEL (bert.model)")
			print("="*40)
			
			lowest_loss_model_loaded = self.load_lowest_loss_model_for_analysis()
			if lowest_loss_model_loaded:
				print(f"Using lowest-loss model from step {self.best_loss_step} (Loss: {self.best_model_loss:.4f}) for analysis")
			else:
				print("Using final training model for analysis")
			
			# Final evaluation with logits for low-confidence analysis
			final_eval_pred, final_eval_loss, final_eval_logits = self._eval_iteration(self.test_data, return_logits=True)
			final_metrics = self._multi_class_metrics(final_eval_pred["pred_ctype_label"], final_eval_pred["ctype_label"], class_names)
			
			# Print comprehensive evaluation summary
			print(f"\nLOWEST LOSS MODEL EVALUATION (Step {self.best_loss_step if lowest_loss_model_loaded else self.step}):")
			self.print_evaluation_summary(final_metrics, class_names)
			
			# Plot confusion matrix for lowest loss model
			self.plot_confusion_matrix(
				final_metrics['confusion_matrix'], 
				class_names, 
				save_path=os.path.join(plots_dir, "confusion_matrix_lowest_loss.png"),
				max_classes=20  # Show top 20 most frequent classes for compactness
			)
			print(f"Lowest loss model confusion matrix saved to: {os.path.join(plots_dir, 'confusion_matrix_lowest_loss.png')}")
			
			# Generate confidence analysis (both low and high confidence)
			self.analyze_confidence_predictions(
				final_eval_pred, 
				final_eval_logits, 
				class_names, 
				plots_dir, 
				low_k=15,
				high_k=15
			)
			
			# Save detailed metrics
			metrics_file = os.path.join(plots_dir, "final_metrics.txt")
			with open(metrics_file, 'w') as f:
				f.write("FINAL EVALUATION METRICS\n")
				f.write("="*50 + "\n")
				f.write(f"Overall Accuracy: {final_metrics['accuracy']:.4f}\n")
				f.write(f"Macro F1-Score: {final_metrics['f1_macro']:.4f}\n")
				f.write(f"Weighted F1-Score: {final_metrics['f1_weighted']:.4f}\n")
				f.write(f"Final Loss: {final_eval_loss:.4f}\n\n")
				
				f.write("PER-CLASS ACCURACY\n")
				f.write("-"*30 + "\n")
				if class_names:
					for i, (name, acc) in enumerate(zip(class_names, final_metrics['per_class_accuracy'])):
						f.write(f"{name:<25}: {acc:.4f}\n")
				
				f.write("\nCLASSIFICATION REPORT\n")
				f.write("-"*30 + "\n")
				f.write(classification_report(final_eval_pred["ctype_label"], final_eval_pred["pred_ctype_label"], 
											target_names=class_names if class_names else None))
			
			print(f"\nFinal evaluation plots and metrics saved to: {plots_dir}")
			
			# Note: Final analysis was performed using the lowest-loss model from bert.model/
			# The best F1-score model is saved separately in bert_best.model/
			print(f"\nFinal analysis completed using lowest-loss model from step {self.best_loss_step}")
			print(f"Best F1-score model (step {self.best_model_step}, Macro-F1 Score {self.best_model_f1_score:.4f}) is saved in bert_best.model/")
			print(f"Lowest-loss model (step {self.best_loss_step}, Loss: {self.best_model_loss:.4f}) is saved in bert.model/")
			
			print("="*60)

	def save(self, file_path: str="output/bert_trained.model"):
		'''
		Save the MethylBERT model in the given path
		'''
		self.bert.to("cpu")
		self.bert.save_pretrained(file_path)

		if hasattr(self.bert, "read_classifier"):
			torch.save(self.bert.read_classifier.state_dict(), os.path.dirname(file_path)+"/read_classification_model.pickle")

		if hasattr(self.bert, "dmr_encoder"):
			torch.save(self.bert.dmr_encoder.state_dict(), os.path.dirname(file_path)+"/dmr_encoder.pickle")

		self.bert.to(self.device)
		print("Step:%d Model Saved on:" % self.step, file_path)

	def load(self, dir_path: str, n_dmrs: int=None, load_fine_tune: bool=False):
		'''
		Load pre-trained / fine-tuned MethylBERT model
		dir_path: str
			Directory to the saved bert model. It must contain "config.json" and "pytorch_model.bin" files
		n_dmrs: int (default: None)
			Number of DMRs to reconstruct the MethylBERT model. If the number is not given, the trainer auto-calculates the number from the same data
		load_fine_tune: bool (default: False)
			Whether the loaded model is a fine-tuned model including num_dmrs or a pre-trained model without num_dmrs
		'''
		print(f"Restore the pretrained model {dir_path}")

		if load_fine_tune:
			'''
			if n_dmrs is not None:
				raise ValueError("You cannot give a new number of DMRs for loading a fine-tuned model. The model should contains one. Please set either n_dmrs=None or load_fine_tune=False")
			'''

			# Get actual number of classes from dataset
			actual_num_classes = self.train_data.dataset.num_classes()
			print(f"Loading fine-tuned model with {actual_num_classes} classes (from dataset)")
			
			self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path,
				output_attentions=False,
				output_hidden_states=False,
				seq_len = self.train_data.dataset.seq_len,
				loss=self._config.loss,
				num_labels=actual_num_classes,  # Number of classification classes (HuggingFace convention)
				num_classes=actual_num_classes,  # Number of classification classes
				num_dmr_embeddings=num_dmrs,  # DMR embedding size
				# Add proper label mappings
				id2label={i: f"CELL_TYPE_{i}" for i in range(actual_num_classes)},
				label2id={f"CELL_TYPE_{i}": i for i in range(actual_num_classes)}
				)

			try:
				self.bert.from_pretrained_dmr_encoder(os.path.dirname(dir_path)+"/dmr_encoder.pickle", self.device)
				print("Restore DMR encoder from %s"%(os.path.dirname(dir_path)+"/dmr_encoder.pickle"))
			except FileNotFoundError:
				print(os.path.dirname(dir_path)+"/dmr_encoder.pickle is not found.")

			try:
				self.bert.from_pretrained_read_classifier(os.path.dirname(dir_path)+"/read_classification_model.pickle", self.device)
				print("Restore read classification FCN model from %s"%(os.path.dirname(dir_path)+"/read_classification_model.pickle"))
			except FileNotFoundError:
				print(os.path.dirname(dir_path)+"/read_classification_model.pickle is not found.")
		else:
			# Get actual number of classes from dataset (needed for both paths)
			actual_num_classes = self.train_data.dataset.num_classes()
			
			# For decoupled focal loss, we need to handle the config properly
			if self._config.loss == 'decoupled_focal':
				print("Using decoupled focal loss - loading with config override")
				print(f"Loading for decoupled focal loss with {actual_num_classes} classes (from dataset)")
				
				# Load the pretrained model but with explicit loss override
				# This bypasses the config validation in the model __init__
				from methylbert.config import MethylBERTConfig
				
				# First, load the pretrained config 
				config = MethylBERTConfig.from_pretrained(dir_path,
					num_labels=actual_num_classes,  # This should be num_classes, not num_dmrs for transformer compatibility
					output_attentions=False,
					output_hidden_states=False,
					seq_len = self.train_data.dataset.seq_len,
					num_classes=actual_num_classes,
					# Add proper label mappings to avoid warnings
					id2label={i: f"CELL_TYPE_{i}" for i in range(actual_num_classes)},
					label2id={f"CELL_TYPE_{i}": i for i in range(actual_num_classes)}
				)
				
				# Override the loss configuration and focal parameters
				config.loss = self._config.loss
				config.focal_gamma = getattr(self._config, 'focal_gamma', 2.0)
				
				# Set DMR-specific parameters
				config.num_dmr_labels = self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs
				
				# Configure decoupled focal loss parameters
				# Use the actual number of classes from dataset
				n_classes = actual_num_classes
				
				alpha_pos_cfg = getattr(self._config, 'focal_alpha_pos', None)
				alpha_neg_cfg = getattr(self._config, 'focal_alpha_neg', None)
				
				alpha_pos_to_use = None
				alpha_neg_to_use = None
				
				# Handle alpha_pos
				if isinstance(alpha_pos_cfg, (list, tuple)):
					if n_classes is not None and len(alpha_pos_cfg) == n_classes:
						alpha_pos_to_use = [float(x) for x in alpha_pos_cfg]
						print(f"Using config-provided per-class focal alpha_pos (len={len(alpha_pos_to_use)}). First 5: {alpha_pos_to_use[:5]}")
					else:
						print(f"Warning: focal_alpha_pos length {len(alpha_pos_cfg)} does not match num classes {n_classes}. Computing automatically.")
						alpha_pos_to_use = self._compute_decoupled_alpha_pos()
				elif isinstance(alpha_pos_cfg, (int, float)):
					alpha_pos_to_use = float(alpha_pos_cfg)
					print(f"Using config-provided scalar focal alpha_pos: {alpha_pos_to_use}")
				else:
					alpha_pos_to_use = self._compute_decoupled_alpha_pos()
					if alpha_pos_to_use is not None:
						print(f"Using auto-computed per-class focal alpha_pos (len={len(alpha_pos_to_use)}). First 5: {alpha_pos_to_use[:5]}")
				
				# Handle alpha_neg  
				if isinstance(alpha_neg_cfg, (list, tuple)):
					if n_classes is not None and len(alpha_neg_cfg) == n_classes:
						alpha_neg_to_use = [float(x) for x in alpha_neg_cfg]
						print(f"Using config-provided per-class focal alpha_neg (len={len(alpha_neg_to_use)}). First 5: {alpha_neg_to_use[:5]}")
					else:
						print(f"Warning: focal_alpha_neg length {len(alpha_neg_cfg)} does not match num classes {n_classes}. Computing automatically.")
						alpha_neg_to_use = self._compute_decoupled_alpha_neg()
				elif isinstance(alpha_neg_cfg, (int, float)):
					alpha_neg_to_use = float(alpha_neg_cfg)
					print(f"Using config-provided scalar focal alpha_neg: {alpha_neg_to_use}")
				else:
					alpha_neg_to_use = self._compute_decoupled_alpha_neg()
					if alpha_neg_to_use is not None:
						print(f"Using auto-computed per-class focal alpha_neg (len={len(alpha_neg_to_use)}). First 5: {alpha_neg_to_use[:5]}")
				
				config.focal_alpha_pos = alpha_pos_to_use
				config.focal_alpha_neg = alpha_neg_to_use
				
				# Set class_weights to None for decoupled focal loss (we use alpha values instead)
				class_weights = None
				
				# Get number of DMRs from dataset for embedding layer
				num_dmrs = self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs
				
				# Now create the model with the modified config
				# Force safetensors usage to avoid torch.load security vulnerability
				import os as os_env
				original_prefer_safetensors = os_env.environ.get('HF_HUB_PREFER_SAFETENSORS', None)
				try:
					# Force safetensors usage
					os_env.environ['HF_HUB_PREFER_SAFETENSORS'] = '1'
					print("Forcing safetensors usage to avoid torch.load security vulnerability")
					
					self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path,
						config=config,
						num_labels=actual_num_classes,  # Number of classification classes (HuggingFace convention)
						num_classes=actual_num_classes,  # Number of classification classes
						num_dmr_embeddings=num_dmrs,  # DMR embedding size
						output_attentions=False,
						output_hidden_states=False,
						seq_len = self.train_data.dataset.seq_len,
						loss=self._config.loss,
						use_safetensors=True,
						# Add proper label mappings to avoid warnings
						id2label={i: f"CELL_TYPE_{i}" for i in range(actual_num_classes)},
						label2id={f"CELL_TYPE_{i}": i for i in range(actual_num_classes)}
						)
				finally:
					# Restore original environment variable
					if original_prefer_safetensors is None:
						os_env.environ.pop('HF_HUB_PREFER_SAFETENSORS', None)
					else:
						os_env.environ['HF_HUB_PREFER_SAFETENSORS'] = original_prefer_safetensors
			else:
				# Standard loading for other loss functions
				print(f"Loading with {actual_num_classes} classes (from dataset)")
				
				# Compute class weights when loading a base model for fine-tuning
				class_weights = None
				if self._config.loss == 'cross_entropy' and hasattr(self.train_data.dataset, 'ctype_label_count'):
					class_weights = self._compute_class_weights()
					if class_weights is not None:
						print("Using class-weighted cross entropy. Weights (first 10):", class_weights[:10])

				# Get number of DMRs from dataset for embedding layer
				num_dmrs = self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs
				
				self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path,
					num_labels=actual_num_classes,  # Number of classification classes (HuggingFace convention)
					num_classes=actual_num_classes,  # Number of classification classes
					num_dmr_embeddings=num_dmrs,  # DMR embedding size
					output_attentions=False,
					output_hidden_states=False,
					seq_len = self.train_data.dataset.seq_len,
					loss=self._config.loss,
					use_safetensors=True,
					# Add proper label mappings to avoid warnings
					id2label={i: f"CELL_TYPE_{i}" for i in range(actual_num_classes)},
					label2id={f"CELL_TYPE_{i}": i for i in range(actual_num_classes)}
					)
			# Ensure classifier head matches actual number of classes from dataset
			expected_classes = actual_num_classes  # Use the actual number from dataset
			if getattr(self.bert, 'num_classes', None) != expected_classes:
				print(f"Adjusting classifier head from {getattr(self.bert, 'num_classes', 'unknown')} to {expected_classes} classes")
				self.bert.num_classes = expected_classes
				in_features = self.bert.read_classifier[-1].in_features
				self.bert.read_classifier[-1] = nn.Linear(in_features, expected_classes)
			# Apply weights after loading (setter will rebuild loss)
			if class_weights is not None and hasattr(self.bert, 'set_class_weights'):
				self.bert.set_class_weights(class_weights)

			# --- NEW: propagate focal weight/gamma when using focal loss ---
			if getattr(self._config, 'loss', 'focal') == 'focal':
				weight_cfg = getattr(self._config, 'focal_weight', None)
				gamma_cfg = getattr(self._config, 'focal_gamma', 2.0)
				# Use the actual number of classes from dataset
				n_classes = actual_num_classes

				resolved_weight = None
				if isinstance(weight_cfg, (list, tuple)):
					if n_classes is not None and len(weight_cfg) == n_classes:
						resolved_weight = [float(x) for x in weight_cfg]
					else:
						print(f"Warning: focal_weight length {len(weight_cfg)} does not match num classes {n_classes}. Recomputing automatically from label frequencies.")
						resolved_weight = self._compute_focal_weight()
				elif weight_cfg is not None:
					print(f"Warning: focal_weight must be a list/tuple for per-class weights or None for auto-computation. Got {type(weight_cfg)}. Auto-computing.")
					resolved_weight = self._compute_focal_weight()
				else:
					resolved_weight = self._compute_focal_weight()

				# Attach to config and rebuild loss in model if needed
				if hasattr(self.bert, 'config'):
					setattr(self.bert.config, 'focal_weight', resolved_weight)
					setattr(self.bert.config, 'focal_gamma', float(gamma_cfg))
					# Rebuild the loss function in case the model was already initialized
					if hasattr(self.bert, 'classification_loss_fct') and hasattr(self.bert, '_setup_loss'):
						self.bert.classification_loss_fct = self.bert._setup_loss('focal', verbose=True)

				# Print resolved weight values
				if isinstance(resolved_weight, (list, tuple)):
					print(f"Resolved focal weight (per-class) length={len(resolved_weight)}; values: {resolved_weight}")
				else:
					print(f"Resolved focal weight: {resolved_weight}")

		# After loading, apply configurable dropout and label smoothing
		if hasattr(self, 'bert'):
			if hasattr(self.bert, 'set_dropout'):
				self.bert.set_dropout(getattr(self._config, 'dropout', 0.1))
			if hasattr(self.bert, 'set_label_smoothing'):
				self.bert.set_label_smoothing(getattr(self._config, 'label_smoothing', 0.0))

		self._setup_model()
		# Initialize classifier bias from validation priors after model is set up/loaded
		self._init_classifier_bias_from_validation()

	def read_classification(self, data_loader: DataLoader = None, tokenizer: MethylVocab = None, logit: bool = False):
		'''
		Classify sequencing reads into cell types

		data_loader: torch.utils.data.DataLoader
			DataLoader containing reads to classify. If nothing is given, the trainer tries to assign 'test_data'
		output_dir: str
			Directory to save the result. If nothing is given, the results is saved in 'save_path'
		save_logit: bool (default: False)
			Whether save the calculated classification logits or not
		'''

		if data_loader is None:
			if self.test_data is None:
				ValueError("There is no test_data assigned to the trainer. Please give a DataLoader as an input.")
			else:
				data_loader = self.test_data

		# classification
		res = dict()
		logits = list()
		self.model.eval()

		pbar = tqdm(total=len(data_loader))
		for i, batch in enumerate(data_loader):

			# 0. batch_data will be sent into the device(GPU or cpu)
			data = dict()

			for k, v in batch.items():
				if type(v) != list:
					data[k] = v.to(self.device)
				if k not in res.keys():
					res[k] = v.numpy() if type(v) == torch.Tensor else v
				else:
					res[k] = np.concatenate([res[k], v.numpy() if type(v) == torch.Tensor else v], axis=0)

			with torch.no_grad():
				with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu",
									enabled=self._config.amp):
					attention_mask = (data["dna_seq"] != self.train_data.dataset.vocab.pad_index).long()
					mask_lm_output = self.model.forward(step=0,
											input_ids = data["dna_seq"],
											attention_mask=attention_mask,
											token_type_ids=data["methyl_seq"],
											labels = data["dmr_label"],
											ctype_label=data["ctype_label"])

				if "pred" in res.keys():
					res["pred"] = np.concatenate([res["pred"], np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)], axis=0)
				else:
					res["pred"] = np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)

				if logit:
					logits.append(mask_lm_output["classification_logits"].cpu().detach().numpy())

			del mask_lm_output
			del data

			pbar.update(1)
		pbar.close()

		if logit:
			logits = np.concatenate(logits, axis=0)
		res["dna_seq"]= [get_dna_seq(s, tokenizer) for s in res["dna_seq"]]
		res["methyl_seq"]=["".join([str(mm) for mm in m]) for m in res["methyl_seq"]]

		res = pd.DataFrame(res)

		return res if not logit else res, logits

	def analyze_confidence_predictions(self, eval_pred, eval_logits, class_names, save_path, low_k=15, high_k=15):
		"""
		Analyze and save both low and high confidence predictions for each true cell type.
		
		:param eval_pred: Dictionary with prediction results
		:param eval_logits: Raw logits from model predictions
		:param class_names: List of class names
		:param save_path: Path to save the analysis file
		:param low_k: Number of top low-confidence examples per class
		:param high_k: Number of top high-confidence examples per class
		"""
		import torch.nn.functional as F
		
		if eval_logits is None:
			print("Warning: Logits not available for confidence analysis")
			return
			
		# Convert logits to probabilities
		probabilities = F.softmax(torch.tensor(eval_logits), dim=1).numpy()
		
		# Get true labels and predicted labels
		true_labels = eval_pred["ctype_label"]
		pred_labels = eval_pred["pred_ctype_label"]
		
		# Calculate confidence for each prediction (max probability)
		confidences = np.max(probabilities, axis=1)
		
		# Analyze both low and high confidence predictions
		low_confidence_analysis = {}
		high_confidence_analysis = {}
		
		for true_class_idx in range(len(class_names)):
			# Find all samples that truly belong to this class
			class_mask = (true_labels == true_class_idx)
			
			if np.sum(class_mask) == 0:
				continue  # Skip if no samples for this class
				
			# Get indices for this class
			class_indices = np.where(class_mask)[0]
			
			# Get confidences and probabilities for this class
			class_confidences = confidences[class_mask]
			class_probabilities = probabilities[class_mask]
			class_predictions = pred_labels[class_mask]
			
			# Sort by confidence: lowest first for low confidence, highest first for high confidence
			sorted_indices_low = np.argsort(class_confidences)  # ascending
			sorted_indices_high = np.argsort(class_confidences)[::-1]  # descending
			
			# Collect unique low-confidence examples
			unique_low_examples = []
			seen_low_patterns = set()
			
			for idx in sorted_indices_low:
				prob_vector = class_probabilities[idx]
				confidence = class_confidences[idx]
				predicted_class = class_predictions[idx]
				original_idx = class_indices[idx]
				
				# Create a signature for uniqueness (rounded probabilities)
				prob_signature = tuple(np.round(prob_vector, 3))
				
				if prob_signature not in seen_low_patterns:
					seen_low_patterns.add(prob_signature)
					
					# Get top 4 classes for this prediction
					top_classes = np.argsort(prob_vector)[::-1][:4]
					
					unique_low_examples.append({
						'original_index': original_idx,
						'confidence': confidence,
						'predicted_class': predicted_class,
						'probabilities': prob_vector,
						'top_classes': top_classes
					})
					
					if len(unique_low_examples) >= low_k:
						break
			
			# Collect unique high-confidence examples
			unique_high_examples = []
			seen_high_patterns = set()
			
			for idx in sorted_indices_high:
				prob_vector = class_probabilities[idx]
				confidence = class_confidences[idx]
				predicted_class = class_predictions[idx]
				original_idx = class_indices[idx]
				
				# Create a signature for uniqueness (rounded probabilities)
				prob_signature = tuple(np.round(prob_vector, 3))
				
				if prob_signature not in seen_high_patterns:
					seen_high_patterns.add(prob_signature)
					
					# Get top 4 classes for this prediction
					top_classes = np.argsort(prob_vector)[::-1][:4]
					
					unique_high_examples.append({
						'original_index': original_idx,
						'confidence': confidence,
						'predicted_class': predicted_class,
						'probabilities': prob_vector,
						'top_classes': top_classes
					})
					
					if len(unique_high_examples) >= high_k:
						break
			
			low_confidence_analysis[true_class_idx] = unique_low_examples
			high_confidence_analysis[true_class_idx] = unique_high_examples
		
		# Save analysis to file
		analysis_file = os.path.join(save_path, "confidence_analysis.txt")
		with open(analysis_file, 'w') as f:
			f.write("CONFIDENCE PREDICTION ANALYSIS\n")
			f.write("="*80 + "\n")
			f.write(f"Analysis of low confidence (top {low_k}) and high confidence (top {high_k}) predictions per true cell type\n")
			f.write("="*80 + "\n\n")
			
			# LOW CONFIDENCE SECTION
			f.write("LOW CONFIDENCE PREDICTIONS\n")
			f.write("="*80 + "\n")
			f.write(f"Top {low_k} unique lowest confidence predictions per true cell type\n")
			f.write("="*80 + "\n\n")
			
			for true_class_idx, examples in low_confidence_analysis.items():
				if not examples:
					continue
					
				true_class_name = class_names[true_class_idx] if class_names else f"Class_{true_class_idx}"
				f.write(f"TRUE CELL TYPE: {true_class_name}\n")
				f.write("-" * 60 + "\n")
				
				for i, example in enumerate(examples, 1):
					self._write_confidence_example(f, example, true_class_idx, true_class_name, class_names, i, "Low")
				
				f.write("\n" + "="*60 + "\n\n")
			
			# HIGH CONFIDENCE SECTION
			f.write("\n\nHIGH CONFIDENCE PREDICTIONS\n")
			f.write("="*80 + "\n")
			f.write(f"Top {high_k} unique highest confidence predictions per true cell type\n")
			f.write("="*80 + "\n\n")
			
			for true_class_idx, examples in high_confidence_analysis.items():
				if not examples:
					continue
					
				true_class_name = class_names[true_class_idx] if class_names else f"Class_{true_class_idx}"
				f.write(f"TRUE CELL TYPE: {true_class_name}\n")
				f.write("-" * 60 + "\n")
				
				for i, example in enumerate(examples, 1):
					self._write_confidence_example(f, example, true_class_idx, true_class_name, class_names, i, "High")
				
				f.write("\n" + "="*60 + "\n\n")
			
			# SUMMARY STATISTICS
			f.write("\n\nSUMMARY STATISTICS\n")
			f.write("="*80 + "\n")
			
			f.write("LOW CONFIDENCE SUMMARY\n")
			f.write("-" * 30 + "\n")
			for true_class_idx, examples in low_confidence_analysis.items():
				if not examples:
					continue
				true_class_name = class_names[true_class_idx] if class_names else f"Class_{true_class_idx}"
				avg_confidence = np.mean([ex['confidence'] for ex in examples])
				min_confidence = np.min([ex['confidence'] for ex in examples])
				f.write(f"{true_class_name:<25}: {len(examples):2d} examples, ")
				f.write(f"avg conf: {avg_confidence:.1%}, min conf: {min_confidence:.1%}\n")
			
			f.write("\nHIGH CONFIDENCE SUMMARY\n")
			f.write("-" * 30 + "\n")
			for true_class_idx, examples in high_confidence_analysis.items():
				if not examples:
					continue
				true_class_name = class_names[true_class_idx] if class_names else f"Class_{true_class_idx}"
				avg_confidence = np.mean([ex['confidence'] for ex in examples])
				max_confidence = np.max([ex['confidence'] for ex in examples])
				f.write(f"{true_class_name:<25}: {len(examples):2d} examples, ")
				f.write(f"avg conf: {avg_confidence:.1%}, max conf: {max_confidence:.1%}\n")
		
		print(f"Confidence analysis saved to: {analysis_file}")
	
	def _write_confidence_example(self, f, example, true_class_idx, true_class_name, class_names, i, conf_type):
		"""
		Helper method to write a confidence example to the file
		"""
		confidence = example['confidence']
		predicted_class = example['predicted_class']
		predicted_class_name = class_names[predicted_class] if class_names else f"Class_{predicted_class}"
		probabilities = example['probabilities']
		top_classes = example['top_classes']

		# Fetch full read information from the test dataset using zero-based index
		read_info = None
		try:
			if hasattr(self, 'test_data') and hasattr(self.test_data, 'dataset') and hasattr(self.test_data.dataset, 'lines'):
				read_info = self.test_data.dataset.lines[example['original_index']]
		except Exception:
			read_info = None

		# Header line for this example
		f.write(f"\n{i:2d}. Confidence: {confidence:.1%} | Predicted: {predicted_class_name}\n")

		# Print full read information if available
		if isinstance(read_info, dict):
			dna_seq_str = read_info.get('dna_seq', '')
			methyl_seq_str = read_info.get('methyl_seq', '')
			dmr_label_val = read_info.get('dmr_label', '')
			ctype_str = read_info.get('ctype', '')
			f.write("    Read Information:\n")
			f.write(f"      dna_seq: {dna_seq_str}\n")
			f.write(f"      methyl_seq: {methyl_seq_str}\n")
			f.write(f"      dmr_label: {dmr_label_val}\n")
			f.write(f"      ctype: {ctype_str}\n")
		else:
			# Fallback if dataset read cannot be fetched
			f.write("    Read Information: (unavailable)\n")

		f.write("    Probability Distribution:\n")
		for rank, class_idx in enumerate(top_classes):
			class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
			prob = probabilities[class_idx]
			marker = " ✓" if class_idx == true_class_idx else ""
			f.write(f"      {prob:5.1%} - {class_name}{marker}\n")

		# Show if the true class is not in top 4
		if true_class_idx not in top_classes:
			true_prob = probabilities[true_class_idx]
			f.write(f"      {true_prob:5.1%} - {true_class_name} ✓ (not in top 4)\n")
