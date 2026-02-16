"""Unified Continual Learning Training Script.

Supports all 11 CL algorithms across all radar datasets.

Usage:
    python train_cl.py --dataset drc --setting scene --algorithm ewc --epochs 30
    python train_cl.py --dataset ci4r --setting class --algorithm ease --epochs 50
    python train_cl.py --dataset radhar --setting class --algorithm l2p --epochs 30
    python train_cl.py --seeds 42 123 456  # Multiple seeds
"""

import argparse
import os
import sys
import json
import csv
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResNet18, RadarTransformer
from datasets import (MicroDopplerDataset, RadarCubeDataset, CI4RDataset,
                      RadHARVoxelDataset, DIATDataset, get_files_from_folders, split_random,
                      DATA_ROOT)
from cl import CLTrainer, CLEvaluator, compute_cl_metrics, ALGORITHMS
from cl.trainer import make_incremental_model, get_feature_dim


# Dataset configurations
DATASET_CONFIGS = {
    'dmm': {
        'settings': ['scene'],
        'scene': {'tasks': ['scene1', 'scene2', 'scene3'], 'num_classes': 4},
        'model': 'ResNet18_Image',  # 3-channel input with colormap
        'normalize': 'image',
        'input_shape': (3, 224, 224),
    },
    'drc': {
        'settings': ['scene'],
        'scene': {'tasks': ['scene1', 'scene2', 'scene3'], 'num_classes': 4},
        'model': 'RadarTransformer_Linear',  # DRC radar cubes
        'normalize': 'log_zscore',
        'input_shape': (21, 5, 5, 25),
    },
    'ci4r': {
        'settings': ['frequency', 'class', 'mixed'],
        'frequency': {'tasks': ['Xethru', '24GHz', '77GHz'], 'num_classes': 11},
        'class': {'tasks': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]], 'num_classes': 11},
        'mixed': {'tasks': [
            ('Xethru', [0, 1, 2, 3]), ('24GHz', [0, 1, 2, 3]), ('77GHz', [0, 1, 2, 3]),
            ('Xethru', [4, 5, 6, 7]), ('24GHz', [4, 5, 6, 7]), ('77GHz', [4, 5, 6, 7]),
            ('Xethru', [8, 9, 10]), ('24GHz', [8, 9, 10]), ('77GHz', [8, 9, 10])
        ], 'num_classes': 11},
        'model': 'ResNet18_Gray',  # 1-channel grayscale
        'normalize': 'minmax',
        'input_shape': (1, 128, 128),
    },
    'radhar': {
        'settings': ['class'],
        'class': {'tasks': [['boxing', 'jack'], ['jump', 'squats'], ['walk']], 'num_classes': 5},
        'model': 'RadarTransformer_Conv3D',  # Voxel data
        'normalize': 'raw_batchnorm',
        'input_shape': (60, 10, 32, 32),
    },
    'diat': {
        'settings': ['class'],
        'class': {'tasks': [[0, 1], [2, 3], [4, 5]], 'num_classes': 6},
        'model': 'ResNet18_Gray',  # 1-channel grayscale
        'normalize': 'zscore',
        'input_shape': (1, 224, 224),
    },
}

TRANSFORMER_MODELS = ['RadarTransformer_Linear', 'RadarTransformer_Conv3D']


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_base_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_model(model_name, num_classes, device):
    """Create model by name."""
    if model_name == 'ResNet18_Image':
        model = ResNet18(num_classes=num_classes, in_channels=3, pretrained=True)
    elif model_name == 'ResNet18_Gray':
        model = ResNet18(num_classes=num_classes, in_channels=1, pretrained=True)
    elif model_name == 'RadarTransformer_Linear':
        model = RadarTransformer(num_classes=num_classes, spatial_encoder='linear')
    elif model_name == 'RadarTransformer_Conv3D':
        model = RadarTransformer(num_classes=num_classes, spatial_encoder='conv3d')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# =============================================================================
# TASK CREATION FUNCTIONS
# =============================================================================

def create_dmm_task_datasets(task_name, seed=42):
    """Create train/val/test datasets for DMM scene task."""
    folder = os.path.join(DATA_ROOT, 'dmm', task_name)
    files = get_files_from_folders([folder])
    if not files:
        return None, None, None
    train_files, val_files, test_files = split_random(files, seed=seed)
    train_ds = MicroDopplerDataset(train_files, augment=True, normalize='image')
    val_ds = MicroDopplerDataset(val_files, augment=False, normalize='image')
    test_ds = MicroDopplerDataset(test_files, augment=False, normalize='image')
    return train_ds, val_ds, test_ds


def create_drc_task_datasets(task_name, seed=42):
    """Create train/val/test datasets for DRC scene task."""
    folder = os.path.join(DATA_ROOT, 'drc', task_name)
    files = get_files_from_folders([folder])
    if not files:
        return None, None, None
    train_files, val_files, test_files = split_random(files, seed=seed)
    train_ds = RadarCubeDataset(train_files, augment=True, normalize='log_zscore')
    val_ds = RadarCubeDataset(val_files, augment=False, normalize='log_zscore')
    test_ds = RadarCubeDataset(test_files, augment=False, normalize='log_zscore')
    return train_ds, val_ds, test_ds


def create_ci4r_frequency_task_datasets(frequency, seed=42):
    """Create datasets for CI4R frequency-based task."""
    root_dir = os.path.join(DATA_ROOT, 'ci4r')
    full_ds = CI4RDataset(root_dir, frequency=frequency, augment=False, normalize='minmax')
    if len(full_ds) == 0:
        return None, None, None
    n = len(full_ds)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    return train_ds, val_ds, test_ds


def filter_dataset_by_classes(dataset, classes, class_attr='labels'):
    """Filter dataset to only include specified classes."""
    if isinstance(dataset, Subset):
        base_ds = dataset.dataset
        base_indices = dataset.indices
    else:
        base_ds = dataset
        base_indices = list(range(len(base_ds)))

    if hasattr(base_ds, class_attr):
        labels = getattr(base_ds, class_attr)
        filtered_indices = [i for i in base_indices if labels[i] in classes]
    else:
        filtered_indices = base_indices
    return Subset(base_ds, filtered_indices)


def create_ci4r_class_task_datasets(classes, seed=42):
    """Create datasets for CI4R class-based task."""
    root_dir = os.path.join(DATA_ROOT, 'ci4r')
    all_train, all_val, all_test = [], [], []
    for freq in ['Xethru', '24GHz', '77GHz']:
        full_ds = CI4RDataset(root_dir, frequency=freq, augment=False, normalize='minmax')
        if len(full_ds) == 0:
            continue
        n = len(full_ds)
        indices = list(range(n))
        np.random.seed(seed)
        np.random.shuffle(indices)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_sub = Subset(full_ds, indices[:n_train])
        val_sub = Subset(full_ds, indices[n_train:n_train + n_val])
        test_sub = Subset(full_ds, indices[n_train + n_val:])
        all_train.append(filter_dataset_by_classes(train_sub, classes))
        all_val.append(filter_dataset_by_classes(val_sub, classes))
        all_test.append(filter_dataset_by_classes(test_sub, classes))
    if not all_train:
        return None, None, None
    train_ds = ConcatDataset(all_train)
    val_ds = ConcatDataset(all_val)
    test_ds = ConcatDataset(all_test)
    return train_ds, val_ds, test_ds


def create_ci4r_mixed_task_datasets(task_spec, seed=42):
    """Create datasets for CI4R mixed task."""
    freq, classes = task_spec
    root_dir = os.path.join(DATA_ROOT, 'ci4r')
    full_ds = CI4RDataset(root_dir, frequency=freq, augment=False, normalize='minmax')
    if len(full_ds) == 0:
        return None, None, None
    n = len(full_ds)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_sub = Subset(full_ds, indices[:n_train])
    val_sub = Subset(full_ds, indices[n_train:n_train + n_val])
    test_sub = Subset(full_ds, indices[n_train + n_val:])
    train_ds = filter_dataset_by_classes(train_sub, classes)
    val_ds = filter_dataset_by_classes(val_sub, classes)
    test_ds = filter_dataset_by_classes(test_sub, classes)
    return train_ds, val_ds, test_ds


def create_radhar_class_task_datasets(activities, seed=42):
    """Create datasets for RadHAR class-based task."""
    root_dir = os.path.join(DATA_ROOT, 'radhar', 'Web_Radhar_Shared_Dataset')
    activity_map = {a: i for i, a in enumerate(['boxing', 'jack', 'jump', 'squats', 'walk'])}
    class_indices = [activity_map[a] for a in activities if a in activity_map]
    train_ds = RadHARVoxelDataset(root_dir, split='train', augment=True, normalize='raw_batchnorm')
    test_ds = RadHARVoxelDataset(root_dir, split='test', augment=False, normalize='raw_batchnorm')
    if len(train_ds) == 0:
        return None, None, None
    train_filtered = []
    for i in range(len(train_ds)):
        if train_ds.labels[i] in class_indices:
            train_filtered.append(i)
    test_filtered = [i for i in range(len(test_ds)) if test_ds.labels[i] in class_indices]
    np.random.seed(seed)
    np.random.shuffle(train_filtered)
    n_train = int(len(train_filtered) * 0.85)
    train_idx = train_filtered[:n_train]
    val_idx = train_filtered[n_train:]
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx), Subset(test_ds, test_filtered)


def create_diat_class_task_datasets(classes, seed=42):
    """Create datasets for DIAT class-based task."""
    root_dir = os.path.join(DATA_ROOT, 'diat', 'DIAT-RadHAR')
    files, labels = [], []
    for class_name, class_idx in DIATDataset.CLASSES.items():
        if class_idx not in classes:
            continue
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(class_dir, f))
                labels.append(class_idx)
    if not files:
        return None, None, None
    np.random.seed(seed)
    indices = np.random.permutation(len(files))
    n_train = int(len(files) * 0.7)
    n_val = int(len(files) * 0.15)
    train_files = [files[i] for i in indices[:n_train]]
    train_labels = [labels[i] for i in indices[:n_train]]
    val_files = [files[i] for i in indices[n_train:n_train + n_val]]
    val_labels = [labels[i] for i in indices[n_train:n_train + n_val]]
    test_files = [files[i] for i in indices[n_train + n_val:]]
    test_labels = [labels[i] for i in indices[n_train + n_val:]]
    train_ds = DIATDataset(train_files, train_labels, augment=True, normalize='zscore')
    val_ds = DIATDataset(val_files, val_labels, augment=False, normalize='zscore')
    test_ds = DIATDataset(test_files, test_labels, augment=False, normalize='zscore')
    return train_ds, val_ds, test_ds


def create_task_datasets(dataset_name, setting, task, seed=42):
    """Create datasets for a specific task."""
    if dataset_name == 'dmm':
        return create_dmm_task_datasets(task, seed)
    elif dataset_name == 'drc':
        return create_drc_task_datasets(task, seed)
    elif dataset_name == 'ci4r':
        if setting == 'frequency':
            return create_ci4r_frequency_task_datasets(task, seed)
        elif setting == 'class':
            return create_ci4r_class_task_datasets(task, seed)
        elif setting == 'mixed':
            return create_ci4r_mixed_task_datasets(task, seed)
    elif dataset_name == 'radhar':
        return create_radhar_class_task_datasets(task, seed)
    elif dataset_name == 'diat':
        return create_diat_class_task_datasets(task, seed)
    return None, None, None


def get_task_classes(dataset_name, setting, task, num_classes):
    """Extract the class indices for a given task."""
    if setting == 'scene' or setting == 'frequency':
        return list(range(num_classes))
    elif setting == 'class':
        if isinstance(task, (list, tuple)):
            if isinstance(task[0], str):
                activity_map = {'boxing': 0, 'jack': 1, 'jump': 2, 'squats': 3, 'walk': 4}
                return [activity_map[a] for a in task if a in activity_map]
            else:
                return list(task)
        return [task]
    elif setting == 'mixed':
        if isinstance(task, tuple) and len(task) == 2:
            _, classes = task
            return list(classes)
        return list(range(num_classes))
    return list(range(num_classes))


def is_class_incremental_setting(setting):
    """Check if the setting is class-incremental."""
    return setting in ['class', 'mixed']


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(dataset_name, setting, algorithm, epochs, seed, device, log_dir, buffer_ratio=0.1):
    """Run benchmark for a specific dataset/setting/algorithm combination."""
    config = DATASET_CONFIGS[dataset_name]
    model_name = config['model']

    # Skip EASE for non-class-incremental settings
    if algorithm == 'ease' and setting not in ['class', 'mixed']:
        print(f"\nSkipping EASE for {dataset_name}/{setting} - EASE is designed for class-incremental only")
        return None

    # Skip prompt-based methods for CNN models (EPB gracefully degrades to HEC-only for CNNs)
    if algorithm in ['l2p', 'coda', 'dualprompt'] and model_name not in TRANSFORMER_MODELS:
        print(f"\nSkipping {algorithm} for {dataset_name} - {algorithm} requires transformer model, got {model_name}")
        return None

    set_seed(seed)
    setting_config = config[setting]
    tasks = setting_config['tasks']
    num_classes = setting_config['num_classes']
    is_class_incr = is_class_incremental_setting(setting)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Setting: {setting}, Algorithm: {algorithm}")
    print(f"Model: {model_name}, Normalize: {config.get('normalize', 'default')}")
    print(f"Tasks: {len(tasks)}, Seed: {seed}")
    if is_class_incr:
        print(f"Mode: Class-Incremental (with output masking)")
    print(f"{'='*60}")

    # Create model
    if is_class_incr:
        model = create_model(model_name, num_classes, device)
        if algorithm != 'ease':
            model = make_incremental_model(model, device)
    else:
        model = create_model(model_name, num_classes, device)

    # Create dataloaders for all tasks
    task_data = []
    for task in tasks:
        train_ds, val_ds, test_ds = create_task_datasets(dataset_name, setting, task, seed)
        if train_ds is None or len(train_ds) == 0:
            print(f"  Warning: No data for task {task}")
            continue
        task_classes = get_task_classes(dataset_name, setting, task, num_classes)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        task_data.append({
            'name': task,
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'classes': task_classes
        })

    if not task_data:
        print("  No valid tasks found!")
        return None

    # Calculate buffer size
    total_train_samples = sum(len(t['train'].dataset) for t in task_data)
    buffer_size = max(10, int(buffer_ratio * total_train_samples))
    if algorithm in ['replay', 'derpp']:
        print(f"Buffer: {buffer_ratio*100:.0f}% of {total_train_samples} samples = {buffer_size}")

    # Initialize trainer and evaluator
    n_tasks = len(task_data) if task_data else len(tasks)
    cl_config = {
        'buffer_size': buffer_size,
        'ewc_importance': 1000,
        'lwf_temperature': 2.0,
        'lwf_alpha': 1.0,
        'derpp_alpha': 0.5,
        'derpp_beta': 0.5,
        'proj_dim': 128,
        'temperature': 0.5,
        'bottleneck_dim': 32,
        'ease_alpha': 0.1,
        'pool_size': 20 if algorithm in ['l2p', 'epb'] else 100,
        'prompt_length': 5 if algorithm in ['l2p', 'dualprompt', 'epb'] else 8,
        'top_k': 5,
        'ortho_weight': 0.1,
        'g_prompt_length': 5,
        'e_pool_size': 10,
        'n_tasks': n_tasks,
        # EPB hyperparameters
        'epb_prompt_type': 'l2p',
        'use_hec': True,
        'use_pcf': True,
        'use_fal': False,
        'epb_ewc_lambda': 500,
        'epb_fisher_ema': 0.7,
        'num_anchors_per_class': 10,
        'anchor_margin': 0.5,
        'fal_lambda': 0.1,
        'epb_use_replay': False,
    }

    trainer = CLTrainer(model, algorithm, device, cl_config)
    evaluator = CLEvaluator(len(task_data))

    # Continual learning loop
    for task_idx, task_info in enumerate(task_data):
        print(f"\n  Task {task_idx + 1}/{len(task_data)}: {task_info['name']}")

        task_classes = task_info['classes']

        # Set model-specific learning rate (transformers need lower lr)
        lr = 1e-4 if model_name in TRANSFORMER_MODELS else 1e-3

        # Train on current task
        val_acc = trainer.train_task(
            task_idx, task_info['train'], task_info['val'],
            task_classes, epochs=epochs, lr=lr
        )
        print(f"    Val Acc: {val_acc:.2f}%")

        # Update CL method after task
        trainer.after_task(task_info['train'], task_classes)

        # Get modules for evaluation
        ease = trainer.cl_params.get('ease') if algorithm == 'ease' else None
        prompt_module = None
        prompt_method = None
        if algorithm in ['l2p', 'coda', 'dualprompt']:
            prompt_method = algorithm
            prompt_module = trainer.cl_params.get(algorithm)
        elif algorithm == 'epb' and 'epb' in trainer.cl_params:
            prompt_method = 'epb'
            prompt_module = trainer.cl_params['epb'].prompt_method

        # Evaluate on all tasks seen so far
        for j in range(task_idx + 1):
            prev_task_classes = task_data[j]['classes'] if is_class_incr else None
            test_acc = evaluator.evaluate_task(
                model, task_data[j]['test'], task_idx, j, device,
                task_classes=prev_task_classes, ease=ease,
                prompt_module=prompt_module, prompt_method=prompt_method
            )
            print(f"    Test Acc (Task {j + 1}): {test_acc:.2f}%")

    # Compute metrics
    metrics = evaluator.get_metrics()
    acc_matrix = evaluator.get_accuracy_matrix()

    print(f"\n  Final Results:")
    print(f"    Final Acc: {metrics['final_acc']:.4f}")
    print(f"    Avg Acc: {metrics['avg_acc']:.4f}")
    print(f"    Forgetting: {metrics['forgetting']:.4f}")

    # Save log
    log_file = os.path.join(log_dir, f"{dataset_name}_{setting}_{algorithm}_seed{seed}.json")
    result = {
        'dataset': dataset_name,
        'setting': setting,
        'algorithm': algorithm,
        'seed': seed,
        'num_tasks': len(task_data),
        'buffer_ratio': buffer_ratio,
        'buffer_size': buffer_size,
        'acc_matrix': acc_matrix[:len(task_data), :len(task_data)].tolist(),
        **metrics
    }

    with open(log_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Training')
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default=None, choices=ALGORITHMS)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--buffer_ratio', type=float, default=0.1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.join(get_base_path(), 'results', 'cl')
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Determine which experiments to run
    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    algorithms = [args.algorithm] if args.algorithm else ALGORITHMS

    all_results = []

    for dataset_name in datasets:
        config = DATASET_CONFIGS[dataset_name]
        settings = [args.setting] if args.setting else config['settings']

        for setting in settings:
            if setting not in config['settings']:
                print(f"Warning: Setting '{setting}' not valid for {dataset_name}, skipping")
                continue

            for algorithm in algorithms:
                for seed in args.seeds:
                    result = run_benchmark(
                        dataset_name, setting, algorithm,
                        args.epochs, seed, args.device, log_dir,
                        buffer_ratio=args.buffer_ratio
                    )
                    if result:
                        all_results.append(result)

    # Save summary results
    if all_results:
        csv_file = os.path.join(results_dir, 'cl_results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'setting', 'algorithm', 'seed', 'num_tasks',
                'buffer_ratio', 'buffer_size',
                'final_acc', 'avg_acc', 'forgetting', 'fwd_transfer'
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in writer.fieldnames})

        json_file = os.path.join(results_dir, 'cl_results.json')
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Results saved to: {results_dir}")
        print(f"Total runs: {len(all_results)}")


if __name__ == '__main__':
    main()
