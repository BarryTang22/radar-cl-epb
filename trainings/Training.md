# Training Commands

## DMM (Micro-Doppler)
```bash
python trainings/train_plain_classification.py --dataset dmm --scene all
python trainings/train_plain_classification.py --dataset dmm --scene 1
python trainings/train_plain_classification.py --dataset dmm --scene 2
python trainings/train_plain_classification.py --dataset dmm --scene 3
```

## DRC (Radar Cube)
```bash
python trainings/train_plain_classification.py --dataset drc --scene all
python trainings/train_plain_classification.py --dataset drc --scene 1
python trainings/train_plain_classification.py --dataset drc --scene 2
python trainings/train_plain_classification.py --dataset drc --scene 3
```

## CI4R (Multi-frequency)
```bash
python trainings/train_plain_classification.py --dataset ci4r --frequency 77GHz
python trainings/train_plain_classification.py --dataset ci4r --frequency 24GHz
python trainings/train_plain_classification.py --dataset ci4r --frequency Xethru
```

## RadHAR (Voxels)
```bash
python trainings/train_plain_classification.py --dataset radhar
```

## DIAT
```bash
python trainings/train_plain_classification.py --dataset diat
```

## Custom Examples

### Change epochs
```bash
python trainings/train_plain_classification.py --dataset dmm --epochs 100
```

### Change batch size and learning rate
```bash
python trainings/train_plain_classification.py --dataset drc --batch_size 16 --lr 5e-4
```

### Specify output directory
```bash
python trainings/train_plain_classification.py --dataset ci4r --output_dir ./results/ci4r
```

### Override data root
```bash
python trainings/train_plain_classification.py --dataset radhar --data_root /path/to/radhar
```
