# ECG Foundation Model with SimCLR and Hybrid Architectures

A PyTorch implementation for self-supervised pretraining and downstream finetuning of ECG analysis models using contrastive learning and hybrid CNN-Transformer architectures.

## Key Features
- **SimCLR Framework**: Contrastive pretraining with NT-Xent loss
- **Multi-Architecture Support**: ResNet1D, CNN-Transformer, and hybrid models
- **HPC Integration**: SLURM scripts for GPU cluster deployment
- **Configuration Hierarchy**: YAML-based config management with default merging
- **Flexible Pipelines**: Separate pretraining/finetuning workflows with metric tracking

## Project Structure

### Core Modules
| File | Description |
|------|-------------|
| **`support_model.py`** | Optimizer management (Adam/AdamW) with warmup+cosine annealing<br>Model save/load utilities |
| **`support_dataset.py`** | ECG data pipelines:<br>- Pretraining/finetuning datasets<br>- Normalization & augmentation<br>- Custom DataLoader setup |
| **`support_based.py`** | Metrics (AUC/Accuracy), one-hot encoding,<br>result serialization with pickle |
| **`support_args.py`** | Unified CLI arguments:<br>- Model arch, dataset params<br>- Training schedules, HPC configs |

### Model Architectures
| File | Description |
|------|-------------|
| **`support_resnet1d.py`** | 1D Residual Network with depth configurations |
| **`support_cnn_transformer.py`** | Hybrid CNN-Transformer for sequential feature learning |
| **`simclr.py`** | SimCLR_encoder implementation with projection head |

### Training Workflows
| File | Purpose |
|------|---------|
| **`model_run.py`** | Main pretraining/finetuning controller |
| **`model_optimization.py`** | Training loops, loss calculations, eval metrics |
| **`main_pretrain.py`** | Launch SimCLR contrastive pretraining |
| **`main_finetune.py`** | Start downstream task fine-tuning |

### Analysis & Results
| File | Function |
|------|----------|
| **`AA99_01_test.py`** | Data subset extraction & preparation |
| **`AA99_02_test.py`** | Training result aggregation/analysis |
| **`AA01_01_read_results.py`** | Performance metric consolidation |

### HPC Integration
| Script | Purpose |
|--------|---------|
| **`main_pretrain.sbatch`** | GPU job for contrastive pretraining |
| **`main_finetune.sbatch`** | GPU job for classification fine-tuning |

### Configuration
| File | Role |
|------|-----|
| **`yaml_config_hook.py`** | Hierarchical YAML config loader with default merging |
