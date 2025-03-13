import numpy as np
import random

fi_na = '/share/home/px/Project/Project20_ECG_foundation_model/Data/pretrain_dataset/Hosp1_data.npy'
da = np.load(fi_na)
print(da.shape)

index = np.arange(da.shape[0])
print(index[:10])

random.shuffle(index)
print(index[:10])

da = da[index[:1000]]

np.save('/share/home/px/Project/Project20_ECG_foundation_model/Data/pretrain_dataset/Hosp1_data_small.npy', da)
