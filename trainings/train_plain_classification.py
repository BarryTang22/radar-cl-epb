"""Training script for plain classification on radar datasets.

Supports all 5 radar modalities: DMM, DRC, CI4R, RadHAR, DIAT.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import (
    create_dataloaders, create_ci4r_dataloaders,
    create_radhar_dataloaders, create_diat_dataloaders, DATA_ROOT
)
from models import ResNet18, RadarTransformer

DATASET_CONFIGS = {
    'dmm': {
        'model': 'resnet18',
        'in_channels': 3,
        'num_classes': 4,
        'normalize': 'image',
        'lr': 1e-3,
    },
    'drc': {
        'model': 'radartransformer',
        'spatial_encoder': 'linear',
        'input_dim': 25,
        'spatial_shape': (5, 5),
        'num_classes': 4,
        'normalize': 'log_zscore',
        'lr': 1e-4,
    },
    'ci4r': {
        'model': 'resnet18',
        'in_channels': 1,
        'num_classes': 11,
        'normalize': 'minmax',
        'lr': 1e-3,
    },
    'radhar': {
        'model': 'radartransformer',
        'spatial_encoder': 'conv3d',
        'depth_dim': 10,
        'height_dim': 32,
        'width_dim': 32,
        'num_classes': 5,
        'normalize': 'raw_batchnorm',
        'lr': 1e-4,
    },
    'diat': {
        'model': 'resnet18',
        'in_channels': 1,
        'num_classes': 6,
        'normalize': 'zscore',
        'lr': 1e-3,
    },
}


def get_dataloaders(args, config):
    """Create dataloaders based on dataset type."""
    if args.dataset in ['dmm', 'drc']:
        modality = 'mmdrive' if args.dataset == 'dmm' else 'cube'
        return create_dataloaders(
            modality=modality,
            scene=args.scene,
            batch_size=args.batch_size,
            split_mode=args.split_mode,
            normalize=config['normalize'],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
    elif args.dataset == 'ci4r':
        root = args.data_root or os.path.join(DATA_ROOT, 'ci4r')
        return create_ci4r_dataloaders(
            root_dir=root,
            frequency=args.frequency,
            batch_size=args.batch_size,
            normalize=config['normalize']
        )
    elif args.dataset == 'radhar':
        root = args.data_root or os.path.join(DATA_ROOT, 'radhar')
        return create_radhar_dataloaders(
            root_dir=root,
            batch_size=args.batch_size,
            normalize=config['normalize']
        )
    elif args.dataset == 'diat':
        root = args.data_root or os.path.join(DATA_ROOT, 'diat')
        return create_diat_dataloaders(
            root_dir=root,
            batch_size=args.batch_size,
            normalize=config['normalize']
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def get_model(config, device):
    """Create model based on configuration."""
    if config['model'] == 'resnet18':
        model = ResNet18(
            num_classes=config['num_classes'],
            in_channels=config['in_channels'],
            pretrained=True
        )
    elif config['model'] == 'radartransformer':
        if config['spatial_encoder'] == 'linear':
            model = RadarTransformer(
                num_classes=config['num_classes'],
                spatial_encoder='linear',
                input_dim=config['input_dim'],
                spatial_shape=config['spatial_shape'],
            )
        else:
            model = RadarTransformer(
                num_classes=config['num_classes'],
                spatial_encoder='conv3d',
                depth_dim=config['depth_dim'],
                height_dim=config['height_dim'],
                width_dim=config['width_dim'],
            )
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train radar classification models')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dmm', 'drc', 'ci4r', 'radhar', 'diat'],
                        help='Dataset to train on')
    parser.add_argument('--scene', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Scene for DMM/DRC datasets')
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['participant', 'random'],
                        help='Split method for DMM/DRC (participant=leave-participant-out, random=random split)')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='Fraction of data for training (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of data for validation (default: 0.1, test gets the rest)')
    parser.add_argument('--frequency', type=str, default='77GHz',
                        choices=['Xethru', '24GHz', '77GHz'],
                        help='Frequency for CI4R dataset')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config = DATASET_CONFIGS[args.dataset]
    lr = args.lr if args.lr is not None else config['lr']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}, Config: {config}")

    train_loader, val_loader, test_loader = get_dataloaders(args, config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = get_model(config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model']}, Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = f"{args.dataset}"
    if args.dataset in ['dmm', 'drc']:
        exp_name += f"_scene{args.scene}_{args.split_mode}"
    elif args.dataset == 'ci4r':
        exp_name += f"_{args.frequency}"
    checkpoint_path = os.path.join(args.output_dir, f"{exp_name}_best.pt")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, checkpoint_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    results = {
        'dataset': args.dataset,
        'scene': args.scene if args.dataset in ['dmm', 'drc'] else None,
        'split_mode': args.split_mode if args.dataset in ['dmm', 'drc'] else None,
        'frequency': args.frequency if args.dataset == 'ci4r' else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': lr,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history,
    }
    results_path = os.path.join(args.output_dir, f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
