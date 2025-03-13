import os
import torch
import numpy as np
from torch.utils.data import Dataset
from simclr.modules.transformations import Jittering, Scaling, Flipping

def ECG_dataset(args):
    ecg_da, ecg_la = read_ECG(args)

    if args.dataset_type == 'pretrain':
        train_dataset = CustomTensorDataset(data=(ecg_da, ecg_la), transform_A=Jittering(0, 0.1))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        train_dataset = CustomTensorDataset(data=(ecg_da, ecg_la))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logistic_batch_size, shuffle=True, drop_last=False)

    args.n_channel = ecg_da.shape[1]
    args.n_length = ecg_da.shape[2]
    args.n_class = int(np.max(ecg_la) + 1)

    return train_loader

class CustomTensorDataset(Dataset):
    """TensorDataset with support for transformations."""
    def __init__(self, data, transform_A=None, transform_B=None):
        assert all(data[0].shape[0] == item.shape[0] for item in data), "Data tensors must have the same first dimension"
        self.data = data
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __getitem__(self, index):
        x = self.data[0][index]

        x1 = self.transform_A(x) if self.transform_A else x
        x2 = self.transform_B(x) if self.transform_B else x

        y = self.data[1][index]
        return torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(y)

    def __len__(self):
        return self.data[0].shape[0]

def read_ECG(args):
    """Reads ECG data from disk, applies preprocessing, and returns ECG data and labels."""
    da_pa = os.path.join('../../Data', f"{args.dataset_type}_dataset")
    ecg_da = np.load(os.path.join(da_pa, 'fuer_da.npy'))

    ecg_la = np.load(os.path.join(da_pa, 'fuer_lab.npy')) if os.path.isfile(os.path.join(da_pa, 'fuer_lab.npy')) else np.zeros(ecg_da.shape[0])

    # Limit data if running locally (for debugging)
    if args.running_env == 'local':
        ecg_da = ecg_da[:200]
        ecg_la = ecg_la[:200]

    # Apply label ratio for finetuning
    if args.dataset_type == 'finetune':
        le = int(args.labelled_ratio * ecg_da.shape[0])
        ecg_da = ecg_da[:le]
        ecg_la = ecg_la[:le]

    if args.whe_mix_lead == 'mix':
        ecg_la = np.tile(ecg_la, ecg_da.shape[1])  # Replicate labels across leads
        ecg_la = np.reshape(ecg_la, (-1))  # Flatten labels
        ecg_da = np.reshape(ecg_da, (-1, 1, ecg_da.shape[-1]))  # Reshape data

    ecg_da = (ecg_da - np.mean(ecg_da)) / np.std(ecg_da)

    return ecg_da, ecg_la
