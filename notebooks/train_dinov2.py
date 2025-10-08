"""
DINOv2-Small implementation for CIFAR-10
- Uses the official DINOv2-Small model directly from Facebook Research via PyTorch Hub
- Designed to work with limited GPU memory
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Global variables
BATCH_SIZE = 8  # Very small for memory efficiency
EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * GRAD_ACCUMULATION_STEPS

class CIFAR10Dataset:
    """Wrapper for CIFAR-10 dataset with train/val/test splits"""
    def __init__(self, root='./data'):
        # Define transforms for CIFAR-10
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        # Load CIFAR-10 datasets
        print("Loading CIFAR-10 dataset...")
        self.trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=self.transform_train
        )
        self.testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=self.transform_test
        )
        
        # Split training set into train/val
        train_size = 45000
        val_size = 5000
        indices = list(range(len(self.trainset)))
        np.random.shuffle(indices)
        
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:train_size+val_size]
        
        self.trainset_subset = Subset(self.trainset, self.train_indices)
        self.valset_subset = Subset(self.trainset, self.val_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.trainset_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.valset_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.testset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        print(f"Dataset ready: {train_size} training samples, {val_size} validation samples, {len(self.testset)} test samples")

class DINOv2SmallCIFAR10(nn.Module):
    """DINOv2-Small model adapted for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(DINOv2SmallCIFAR10, self).__init__()
        # Load DINOv2-Small model via PyTorch Hub
        print("Loading DINOv2-Small model from PyTorch Hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Get embedding dimension
        self.embed_dim = 384  # DINOv2-Small embedding dimension is 384
        
        # Create a custom classifier for CIFAR-10
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Freeze most layers except the last blocks
        # This is common practice for fine-tuning vision transformers
        for name, param in self.model.named_parameters():
            if 'blocks.11' not in name:  # Only train the last transformer block
                param.requires_grad = False
    
    def forward(self, x):
        # Resize CIFAR-10 images to 224x224 as expected by DINOv2
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get features from DINOv2-Small (extract the CLS token)
        with torch.no_grad():
            features = self.model(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits

def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """Train for one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets) / GRAD_ACCUMULATION_STEPS
        
        # Backward pass with gradient accumulation
        loss.backward()
        
        # Step optimizer every GRAD_ACCUMULATION_STEPS
        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Update running statistics
        running_loss += loss.item() * GRAD_ACCUMULATION_STEPS * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss/total:.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
        
        # Clean up GPU memory
        del inputs, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, split="Validation"):
    """Evaluate model on validation or test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Evaluating ({split})"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Clean up GPU memory
            del inputs, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    eval_loss = running_loss / total
    eval_acc = 100. * correct / total
    
    print(f"{split}: Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.2f}%")
    return eval_loss, eval_acc

def main(args):
    """Main training and evaluation function"""
    # Create dataset and loaders
    dataset = CIFAR10Dataset()
    
    # Create model
    print(f"Creating model on {DEVICE}...")
    model = DINOv2SmallCIFAR10().to(DEVICE)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    # Create optimizer - only optimize the trainable parameters
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    start_epoch = 1
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists('dinov2_latest_checkpoint.pth'):
        print("Loading checkpoint...")
        checkpoint = torch.load('dinov2_latest_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        train_accs = checkpoint['train_accs']
        val_losses = checkpoint['val_losses']
        val_accs = checkpoint['val_accs']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Grad accumulation steps: {GRAD_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION_STEPS}")
    
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, dataset.train_loader, optimizer, criterion, epoch
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Step scheduler
            scheduler.step()
            
            # Evaluate
            val_loss, val_acc = evaluate(
                model, dataset.val_loader, criterion, split="Validation"
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc, best_val_acc)
            
            # Save best model
            if is_best:
                print(f"New best validation accuracy: {val_acc:.2f}%")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'val_losses': val_losses,
                    'val_accs': val_accs
                }, 'dinov2_best.pth')
            
            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }, 'dinov2_latest_checkpoint.pth')
            
            # Report time
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        # Load best model
        checkpoint = torch.load('dinov2_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = evaluate(
            model, dataset.test_loader, criterion, split="Test"
        )
        
        # Update best checkpoint with test accuracy
        checkpoint['test_acc'] = test_acc
        torch.save(checkpoint, 'dinov2_best.pth')
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('dinov2_training_curves.png')
        
        # Save model configuration for reference
        with open('dinov2_config.txt', 'w') as f:
            f.write("DINOv2-Small on CIFAR-10 - Configuration\n")
            f.write("="*50 + "\n\n")
            f.write(f"Base model: DINOv2-Small (dinov2_vits14)\n")
            f.write(f"Input size: 224x224 (resized from 32x32)\n")
            f.write(f"Embedding dimension: {model.embed_dim}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Gradient accumulation steps: {GRAD_ACCUMULATION_STEPS}\n")
            f.write(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION_STEPS}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Weight decay: {WEIGHT_DECAY}\n")
            f.write(f"Training epochs: {EPOCHS}\n")
            f.write(f"Optimizer: AdamW\n")
            f.write(f"Scheduler: CosineAnnealingLR\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)\n")
            f.write(f"Final validation accuracy: {best_val_acc:.2f}%\n")
            f.write(f"Test accuracy: {test_acc:.2f}%\n")
        
        return test_acc, best_val_acc
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        return 0.0, best_val_acc if len(val_accs) > 0 else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv2-Small model on CIFAR-10')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    main(args)
