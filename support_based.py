import pickle
import random
import string
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
import os

warnings.simplefilter('ignore')

def com_mul_str(args):
    str_li = [args.model_type, args.whe_mix_lead, args.epochs, args.batch_size, args.n_length]
    if args.model_type == 'resnet1d':
        str_li.append(args.n_block)
    elif args.model_type == 'cnntransf':
        str_li.append(args.num_layers)

    rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    str_li.append(rand_str)
    return '_'.join(map(str, str_li))

def cal_met(output, y, args):
    predicted = output.argmax(1)
    one_hot_y = conv_one_hot(y, args.n_class)

    acc = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, average='macro')
    recall = recall_score(y, predicted, average='macro')
    F1 = f1_score(y, predicted, average='macro')

    try:
        auc = roc_auc_score(one_hot_y, output, average="macro", multi_class="ovr")
    except ValueError:  # Catching specific error for invalid AUC computation
        auc = 0.0

    prc = average_precision_score(one_hot_y, output, average="macro")
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': F1,
        'auc': auc,
        'average_precision': prc
    }

def conv_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def save_res(sa_fo, na, res):
    file_path = os.path.join(sa_fo, na)
    with open(file_path, 'wb') as file:
        pickle.dump(res, file)

def read_res(sa_fo, na):
    file_path = os.path.join(sa_fo, na)
    with open(file_path, 'rb') as file:
        return pickle.load(file)
