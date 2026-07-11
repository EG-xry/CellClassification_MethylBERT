#!/usr/bin/env python3
"""
Diagnostic script to demonstrate focal loss behavior with different gamma values.
This helps understand why loss increases while F1 improves with gamma=0.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def softmax_focal_loss(inputs, targets, weight=None, gamma=2.0, reduction="mean"):
    """Simplified focal loss implementation for demonstration"""
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
    
    return focal_loss.mean() if reduction == "mean" else focal_loss

def demonstrate_focal_loss_behavior():
    """Demonstrate how focal loss behaves with different gamma values"""
    
    # Simulate imbalanced dataset: 3 classes with different frequencies
    # Class 0: 70% of samples (common)
    # Class 1: 20% of samples (moderate) 
    # Class 2: 10% of samples (rare)
    
    print("=== Focal Loss Behavior Analysis ===\n")
    
    # Create synthetic logits and targets
    batch_size = 1000
    num_classes = 3
    
    # Simulate model predictions that get better over time
    # Initially: random predictions
    # Later: more confident but still imbalanced
    
    scenarios = [
        ("Random predictions", 0.0),
        ("Slightly confident", 0.3),
        ("Moderately confident", 0.6),
        ("Very confident", 0.9)
    ]
    
    gamma_values = [0.0, 1.0, 2.0, 3.0]
    
    print("Class distribution: Class 0: 70%, Class 1: 20%, Class 2: 10%")
    print("Equal weights: [1.0, 1.0, 1.0]")
    print("\n" + "="*80)
    
    for scenario_name, confidence in scenarios:
        print(f"\n{scenario_name} (confidence={confidence}):")
        print("-" * 50)
        
        # Create targets with imbalanced distribution
        targets = torch.cat([
            torch.zeros(int(0.7 * batch_size)),  # Class 0: 70%
            torch.ones(int(0.2 * batch_size)),   # Class 1: 20%
            torch.full((int(0.1 * batch_size),), 2)  # Class 2: 10%
        ]).long()
        
        # Create logits that become more confident over time
        logits = torch.randn(batch_size, num_classes)
        
        # Make predictions more confident for the true class
        for i, target in enumerate(targets):
            # Add confidence to the true class
            logits[i, target] += confidence * 2.0
            
            # Add some noise to other classes
            other_classes = [j for j in range(num_classes) if j != target]
            for j in other_classes:
                logits[i, j] += np.random.normal(0, 0.1)
        
        # Equal weights (no class balancing)
        equal_weights = torch.ones(num_classes)
        
        # Compute losses for different gamma values
        for gamma in gamma_values:
            focal_loss = softmax_focal_loss(logits, targets, weight=equal_weights, gamma=gamma)
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == targets).float().mean()
            
            # Compute per-class accuracy
            class_accuracies = []
            for class_id in range(num_classes):
                class_mask = targets == class_id
                if class_mask.sum() > 0:
                    class_acc = (predictions[class_mask] == targets[class_mask]).float().mean()
                    class_accuracies.append(class_acc.item())
                else:
                    class_accuracies.append(0.0)
            
            # Compute macro F1 (simplified)
            macro_f1 = np.mean(class_accuracies)
            
            print(f"  Gamma={gamma:1.1f}: Loss={focal_loss:.4f}, Acc={accuracy:.4f}, Macro-F1={macro_f1:.4f}")
            print(f"    Per-class acc: Class0={class_accuracies[0]:.3f}, Class1={class_accuracies[1]:.3f}, Class2={class_accuracies[2]:.3f}")

def demonstrate_weighted_vs_unweighted():
    """Compare weighted vs unweighted focal loss"""
    
    print("\n\n=== Weighted vs Unweighted Focal Loss ===\n")
    
    # Same setup as before
    batch_size = 1000
    num_classes = 3
    
    targets = torch.cat([
        torch.zeros(int(0.7 * batch_size)),  # Class 0: 70%
        torch.ones(int(0.2 * batch_size)),   # Class 1: 20%
        torch.full((int(0.1 * batch_size),), 2)  # Class 2: 10%
    ]).long()
    
    # Moderately confident predictions
    logits = torch.randn(batch_size, num_classes)
    for i, target in enumerate(targets):
        logits[i, target] += 0.6 * 2.0
        other_classes = [j for j in range(num_classes) if j != target]
        for j in other_classes:
            logits[i, j] += np.random.normal(0, 0.1)
    
    # Equal weights (no balancing)
    equal_weights = torch.ones(num_classes)
    
    # Inverse frequency weights (proper balancing)
    class_counts = torch.bincount(targets)
    inv_freq_weights = 1.0 / (class_counts.float() + 1e-8)
    inv_freq_weights = inv_freq_weights / inv_freq_weights.mean()  # Normalize to mean 1
    
    print("Class counts:", class_counts.tolist())
    print("Equal weights:", equal_weights.tolist())
    print("Inverse freq weights:", inv_freq_weights.tolist())
    print()
    
    for gamma in [0.0, 2.0]:
        print(f"Gamma = {gamma}:")
        
        # Equal weights
        loss_equal = softmax_focal_loss(logits, targets, weight=equal_weights, gamma=gamma)
        predictions = torch.argmax(logits, dim=1)
        acc_equal = (predictions == targets).float().mean()
        
        # Weighted
        loss_weighted = softmax_focal_loss(logits, targets, weight=inv_freq_weights, gamma=gamma)
        acc_weighted = (predictions == targets).float().mean()  # Same predictions
        
        print(f"  Equal weights:    Loss={loss_equal:.4f}, Acc={acc_equal:.4f}")
        print(f"  Inverse freq:     Loss={loss_weighted:.4f}, Acc={acc_weighted:.4f}")
        print()

if __name__ == "__main__":
    demonstrate_focal_loss_behavior()
    demonstrate_weighted_vs_unweighted()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("1. Gamma=0 makes focal loss identical to cross-entropy")
    print("2. Equal weights (1.0, 1.0, 1.0) don't balance class frequencies")
    print("3. As model becomes more confident, cross-entropy loss increases")
    print("4. But accuracy/F1 can still improve due to better class separation")
    print("5. Use proper inverse-frequency weights for class balancing")
    print("6. Consider gamma > 0 to focus on hard examples")
