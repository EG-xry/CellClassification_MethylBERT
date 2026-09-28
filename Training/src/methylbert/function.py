import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Union, Sequence, Optional


def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Softmax-based focal loss for multiclass classification.
    
    Args:
        inputs (Tensor): Logits of shape (N, C) where N is batch size and C is number of classes.
        targets (Tensor): Class indices of shape (N,) with values in [0, C-1].
        weight (Tensor, optional): Per-class weights of shape (C,). If None, all classes have equal weight.
        gamma (float): Focusing parameter. Higher gamma puts more focus on hard examples.
        reduction (str): Specifies the reduction to apply: 'none', 'mean', or 'sum'.
        
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Validate inputs
    if inputs.dim() != 2:
        raise ValueError(f"Expected inputs to be 2D (batch_size, num_classes), got shape {inputs.shape}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets to be 1D (batch_size,), got shape {targets.shape}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch size mismatch: inputs {inputs.shape[0]} vs targets {targets.shape[0]}")
    
    num_classes = inputs.shape[1]
    if targets.max() >= num_classes:
        raise ValueError(f"Target class {targets.max().item()} is out of range for {num_classes} classes")
    if targets.min() < 0:
        raise ValueError(f"Target classes must be non-negative, got minimum {targets.min().item()}")
    
    # Ensure weight is on the same device as inputs if provided
    if weight is not None and not isinstance(weight, type(None)):
        if not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=inputs.dtype, device=inputs.device)
        else:
            weight = weight.to(device=inputs.device, dtype=inputs.dtype)
        
    # Compute cross-entropy loss without reduction
    ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
    
    # Compute probabilities using softmax
    p = F.softmax(inputs, dim=1)
    
    # Get the probability of the true class for each sample
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Apply focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Compute focal loss
    focal_loss = focal_weight * ce_loss
    
    # Apply reduction
    if reduction == "none":
        return focal_loss
    elif reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Supported: 'none', 'mean', 'sum'")


def sigmoid_focal_loss_decoupled(
    inputs,          # (N, C) logits
    targets,         # (N,) class indices or (N, C) one-hot float
    alpha_pos=None,  # (C,) or scalar; weight on *positives* only
    alpha_neg=None,  # (C,) or scalar; weight on *negatives* only
    gamma=1.0,
    reduction="mean",
):
    """
    Decoupled focal loss that allows separate weighting for positive and negative examples.
    This helps address model collapse where one class dominates predictions.
    
    Args:
        inputs (Tensor): Logits of shape (N, C)
        targets (Tensor): Class indices of shape (N,) or one-hot targets of shape (N, C)
        alpha_pos (float, list, Tensor, optional): Weight for positive examples per class
        alpha_neg (float, list, Tensor, optional): Weight for negative examples per class
        gamma (float): Focal loss gamma parameter
        reduction (str): 'none', 'mean', or 'sum'
    
    Returns:
        Loss tensor with the reduction option applied.
    """
    import torch
    import torch.nn.functional as F
    
    # Convert class indices to one-hot if needed
    if targets.dim() == 1:
        # targets is class indices (N,), convert to one-hot (N, C)
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        targets = targets_one_hot
    
    # Probabilities
    p = torch.sigmoid(inputs)

    # Separate pos/neg focal terms (stable logits form)
    # pos: y * (1-p)^γ * (-log p) = y * (1-p)^γ * F.softplus(-inputs)
    # neg: (1-y) * p^γ * (-log (1-p)) = (1-y) * p^γ * F.softplus(inputs)
    loss_pos = targets * ((1 - p).clamp_min(1e-6) ** gamma) * F.softplus(-inputs)
    loss_neg = (1 - targets) * (p.clamp_min(1e-6) ** gamma) * F.softplus(inputs)

    # Apply decoupled weights
    if alpha_pos is not None:
        if not torch.is_tensor(alpha_pos):
            alpha_pos = torch.tensor(alpha_pos, dtype=inputs.dtype, device=inputs.device)
        if alpha_pos.dim() == 1:
            alpha_pos = alpha_pos.view(1, -1)  # Reshape for broadcasting
        loss_pos = loss_pos * alpha_pos
        
    if alpha_neg is not None:
        if not torch.is_tensor(alpha_neg):
            alpha_neg = torch.tensor(alpha_neg, dtype=inputs.dtype, device=inputs.device)
        if alpha_neg.dim() == 1:
            alpha_neg = alpha_neg.view(1, -1)  # Reshape for broadcasting
        loss_neg = loss_neg * alpha_neg

    loss = loss_pos + loss_neg

    if reduction == "mean":
        return loss.sum(dim=1).mean()   # per-sample sum, then batch mean
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction}' \n Supported reduction modes: 'none', 'mean', 'sum'"
        )


class FocalLoss(_Loss):
    """
    Softmax-based focal loss for multiclass classification.
    
    This implementation uses softmax probabilities and cross-entropy loss as the base,
    then applies the focal loss weighting (1 - p_t)^gamma to focus learning on hard examples.
    """
    
    def __init__(self, 
                 weight: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, 
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_focal_loss(input, target, weight=self.weight, gamma=self.gamma, reduction=self.reduction)


class DecoupledFocalLoss(_Loss):
    """
    Focal loss with separate alpha weights for positive and negative examples.
    Helps prevent model collapse where one class dominates predictions.
    """
    
    def __init__(self, 
                 alpha_pos: Union[float, Sequence[float], torch.Tensor] = None,
                 alpha_neg: Union[float, Sequence[float], torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss_decoupled(
            input, target, 
            alpha_pos=self.alpha_pos, 
            alpha_neg=self.alpha_neg, 
            gamma=self.gamma, 
            reduction=self.reduction
        )