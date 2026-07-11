import os
import time
import warnings
import platform

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
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)
        self.f_train = os.path.join(self.save_path, "train.csv")
        self.f_eval = os.path.join(self.save_path, "eval.csv")

        # Training history for plotting
        self.train_history = {"step": [], "loss": [], "accuracy": [], "lr": []}
        self.eval_history = {"step": [], "loss": [], "accuracy": []}


    def save(self, file_path="output/bert_trained.model"):
        '''
        Saving the current BERT model on file_path

        file_path: str
            model output path which gonna be file_path+"ep%d" % epoch
        '''
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

        if not self._config.eval:
            # Setting the AdamW optimizer with hyper-param
            self.optim = AdamW(self.model.parameters(),
                               lr=self._config.lr, betas=self._config.beta, eps=self._config.eps, weight_decay=self._config.weight_decay)

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
            num_labels=self.train_data.dataset.num_dmrs(),
            output_attentions=True,
            output_hidden_states=True,
            hidden_dropout_prob=0.01,
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
        
        # Per-class accuracy
        cm = confusion_matrix(label, pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Create detailed classification report
        if class_names is not None:
            # Use labels parameter to include all classes, even if not predicted
            labels = list(range(len(class_names)))
            report = classification_report(label, pred, target_names=class_names, labels=labels, output_dict=True)
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
        Plot training and evaluation curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.train_history['step']:
            axes[0, 0].plot(self.train_history['step'], self.train_history['loss'], label='Train Loss', color='blue')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.eval_history['step']:
            axes[0, 1].plot(self.eval_history['step'], self.eval_history['loss'], label='Validation Loss', color='red')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy curves
        if self.train_history['step']:
            axes[1, 0].plot(self.train_history['step'], self.train_history['accuracy'], label='Train Accuracy', color='green')
            axes[1, 0].set_title('Training Accuracy')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.eval_history['step']:
            axes[1, 1].plot(self.eval_history['step'], self.eval_history['accuracy'], label='Validation Accuracy', color='orange')
            axes[1, 1].set_title('Validation Accuracy')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_confusion_matrix(self, cm, class_names, save_path=None, max_classes=None):
        """
        Plot a confusion matrix for multi-class classification with all classes shown.
        Dynamically adjust figure size for clarity and show accuracy numbers on each square.
        
        :param cm: confusion matrix
        :param class_names: list of class names
        :param save_path: path to save the plot
        :param max_classes: (ignored, kept for compatibility)
        """
        n_classes = len(class_names)
        # Dynamically set figure size: 0.5 inch per class, min 10, max 40 inches
        fig_size = max(10, min(0.5 * n_classes, 40))
        plt.figure(figsize=(fig_size, fig_size))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

        # Create heatmap with annotations
        # Always show annotations, but adjust format based on number of classes
        annot_format = '.1f' if n_classes <= 20 else '.0f'  # Show 1 decimal for small matrices, 0 for large ones
        
        # Create annotation matrix - show percentages
        annot_matrix = np.empty_like(cm_normalized, dtype=str)
        for i in range(n_classes):
            for j in range(n_classes):
                if cm_normalized[i, j] > 0:
                    if n_classes <= 20:
                        annot_matrix[i, j] = f'{cm_normalized[i, j]:.1%}'  # Show as percentage with 1 decimal
                    else:
                        annot_matrix[i, j] = f'{cm_normalized[i, j]:.0%}'  # Show as percentage with 0 decimals
                else:
                    annot_matrix[i, j] = ''

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=annot_matrix,  # Always show annotations
            fmt='',  # Use custom annotation matrix
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Accuracy'},
            square=True,  # Make squares actually square
            linewidths=0.5,  # Add grid lines
            linecolor='white'
        )

        plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Cell Type', fontsize=12)
        plt.ylabel('True Cell Type', fontsize=12)
        
        # Adjust font sizes based on number of classes
        if n_classes <= 20:
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
        else:
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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


class MethylBertPretrainTrainer(MethylBertTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def create_model(self, *args, **kwargs):
        config = BertConfig(vocab_size = len(self.train_data.dataset.vocab), *args, **kwargs)
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

        scaler = GradScaler() if self._config.amp else None

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
                    if self.test_data is not None and self.step % self._config.eval_freq == 0 and self.step > 0:

                        test_pred, test_loss = self._eval_iteration(self.test_data)
                        idces = np.where(test_pred["label"]>=0)
                        test_pred_acc = self._acc(test_pred["prediction"][idces[0], idces[1]],
                                                  test_pred["label"][idces[0], idces[1]])

                        with open(self.f_eval, "a") as f_perform:
                            f_perform.write("\t".join([str(self.step), str(test_pred_acc), str(test_loss)]) +"\n")

                        del test_pred

                    if self.step % self._config.log_freq == 0:
                        print("\nTrain Step %d iter - loss : %f / lr : %f"%(self.step, global_step_loss, self.optim.param_groups[0]["lr"]))
                        print(f"Running time for iter = {duration}")

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

    def create_model(self, config_file: str = None):
        '''
        Create a new MethylBERT model from the configuration
        '''

        config = MethylBERTConfig.from_pretrained(config_file,
            num_labels=self.train_data.dataset.num_dmrs(),
            output_attentions=True,
            output_hidden_states=True,
            hidden_dropout_prob=0.01,
            vocab_size = len(self.train_data.dataset.vocab),
            loss=self._config.loss,
            num_classes=getattr(self._config, 'num_classes', 39))

        self.bert = MethylBertEmbeddedDMR(config=config,
                                          seq_len=self.train_data.dataset.seq_len,
                                          num_classes=getattr(self._config, 'num_classes', 39))

        # Initialize the BERT Language Model, with BERT model
        self._setup_model()

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
                    mask_lm_output = self.model.forward(step=self.step,
                                            input_ids = data["dna_seq"],
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
            return predict_res, mean_loss, np.concatenate(return_logits, axis=0)

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

        scaler = GradScaler() if self._config.amp else None

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
                    mask_lm_output = self.model.forward(step=self.step,
                                            input_ids=data["dna_seq"],
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
                scaler.scale(loss).backward(retain_graph=True) if self._config.amp else loss.backward(retain_graph=True)

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

                if (local_step+1) % self._config.eval_freq == 0 or local_step == 0:
                    # Evaluation
                    eval_pred, eval_loss = self._eval_iteration(self.test_data)
                    
                    # Calculate multi-class metrics
                    class_names = list(self.train_data.dataset.int_to_ctype.values()) if hasattr(self.train_data.dataset, 'int_to_ctype') else None
                    eval_metrics = self._multi_class_metrics(eval_pred["pred_ctype_label"], eval_pred["ctype_label"], class_names)
                    eval_acc = eval_metrics['accuracy']

                    # Store evaluation history
                    self.eval_history['step'].append(self.step)
                    self.eval_history['loss'].append(eval_loss)
                    self.eval_history['accuracy'].append(eval_acc)

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
                                
                                print(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
                                print(f"Running time for iter = {duration:.2f}s")
                                print(f"Total elapsed: {elapsed_str} | ETA: {eta_str}")
                                print(f"Validation - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
                            else:
                                print(f"\nTrain Step {self.step} - Loss: {global_step_loss:.4f} / LR: {self.optim.param_groups[0]['lr']:.6f}")
                                print(f"Running time for iter = {duration:.2f}s")
                                print(f"Validation - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")

                    # Only save if new min loss AND (step % 1000 == 0 OR in last 20% of training)
                    if self.min_loss > eval_loss:
                        in_last_20pct = (self.step >= int(0.8 * steps))
                        if (self.step % 1000 == 0) or in_last_20pct:
                            if verbose > 0:
                                print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (self.step, eval_loss, self.min_loss, self.save_path))
                            self.save(self.save_path)
                            self.min_loss = eval_loss

                    # For saving an interim model to track the training
                    if ( type(self._config.save_freq) == int ) and (self.step % self._config.save_freq == 0):
                        step_save_dir=self.save_path.replace("bert.model", "bert.model_step%d"%(self.step))
                        if verbose > 0:
                            print("Step %d: Save an interim model at %s"%(self.step, step_save_dir))
                        if not os.path.exists(step_save_dir):
                            os.mkdir(step_save_dir)
                        self.save(step_save_dir)

                    # Save the step info (step, loss, lr, acc)
                    with open(self.f_train, "a") as f_perform:

                        train_prediction_res["dmr_label"] = np.concatenate(train_prediction_res["dmr_label"],  axis=0)
                        train_prediction_res["pred_ctype_label"] = np.concatenate(train_prediction_res["pred_ctype_label"], axis=0)
                        train_prediction_res["ctype_label"] = np.concatenate(train_prediction_res["ctype_label"],  axis=0)
                        
                        # Calculate training metrics
                        train_metrics = self._multi_class_metrics(train_prediction_res["pred_ctype_label"], train_prediction_res["ctype_label"], class_names)
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
                        eval_loss=f"{eval_loss:.4f}",
                        eval_acc=f"{eval_acc:.3f}",
                        lr=f"{self.optim.param_groups[0]['lr']:.6f}"
                    )
                    # Reset prediction result
                    del train_prediction_res
                    train_prediction_res =  {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}

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
            
            # Final evaluation
            final_eval_pred, final_eval_loss = self._eval_iteration(self.test_data)
            class_names = list(self.train_data.dataset.int_to_ctype.values()) if hasattr(self.train_data.dataset, 'int_to_ctype') else None
            final_metrics = self._multi_class_metrics(final_eval_pred["pred_ctype_label"], final_eval_pred["ctype_label"], class_names)
            
            # Print comprehensive evaluation summary
            self.print_evaluation_summary(final_metrics, class_names)
            
            # Generate plots
            plots_dir = os.path.join(self.save_path, "plots")
            if not os.path.exists(plots_dir):
                os.mkdir(plots_dir)
            
            # Plot training curves
            self.plot_training_curves(save_path=os.path.join(plots_dir, "training_curves.png"))
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                final_metrics['confusion_matrix'], 
                class_names, 
                save_path=os.path.join(plots_dir, "confusion_matrix.png"),
                max_classes=20  # Show top 20 most frequent classes for compactness
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

            self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path,
				output_attentions=True,
                output_hidden_states=True,
                seq_len = self.train_data.dataset.seq_len,
                loss=self._config.loss,
                num_labels=n_dmrs,
                num_classes=getattr(self._config, 'num_classes', 39)
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
            self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path,
                num_labels=self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs,
                output_attentions=True,
                output_hidden_states=True,
                seq_len = self.train_data.dataset.seq_len,
                loss=self._config.loss,
                num_classes=getattr(self._config, 'num_classes', 39)
                )

        self._setup_model()

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
                    mask_lm_output = self.model.forward(step=0,
                                            input_ids = data["dna_seq"],
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
        res["dna_seq"]=[get_dna_seq(s, tokenizer) for s in res["dna_seq"]]
        res["methyl_seq"]=["".join([str(mm) for mm in m]) for m in res["methyl_seq"]]

        res = pd.DataFrame(res)

        return res if not logit else res, logits
