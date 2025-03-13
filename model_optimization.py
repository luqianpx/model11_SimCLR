import torch
import support_based as spb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def process_batch(args, x, x_aug=None, y=None, encoder=None, classifier=None, criterion=None, optimizer=None, mode='train'):
    """Process a single batch of data for training or evaluation"""
    # Move data to the right device
    x = x.to(args.device)
    if y is not None:
        y = y.to(args.device)

    # Forward pass through encoder
    if mode == 'train':
        with torch.no_grad() if args.downstream_tr_type == "pretrain_only_classifier" else torch.enable_grad():
            h, _, z, _ = encoder(x, x)  # Assuming encoder returns h, z
        output = classifier(h)
        loss = criterion(output, y)
        metrics = calculate_metrics(output, y, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elif mode == 'eval':
        with torch.no_grad():
            h, _, z, _ = encoder(x, x)
        output = classifier(h)
        loss = criterion(output, y)
        metrics = calculate_metrics(output, y, args)
    return loss.item(), metrics

def calculate_metrics(output, y, args):
    """Calculate classification metrics"""
    output = output.detach().cpu().numpy()
    y = y.cpu().numpy()
    metrics = spb.cal_met(output, y, args)
    return metrics

def train(args, train_loader, model, criterion, optimizer):
    model.train()
    loss_epoch = []
    for step, (x_i, x_j, _) in enumerate(tqdm(train_loader)):
        loss, _ = process_batch(args, x_i, x_j, mode='train', encoder=model, classifier=model.classifier, criterion=criterion, optimizer=optimizer)
        loss_epoch.append(loss)
        if step % 100 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss}")
    return np.mean(loss_epoch)

def finetune_train(args, loader, encoder, classifier, criterion, optimizer):
    encoder.train()
    classifier.train()
    loss_epoch = []
    met_epoch = []
    for _, (x, _, y) in enumerate(loader):
        loss, metrics = process_batch(args, x, x_aug=None, y=y, encoder=encoder, classifier=classifier, criterion=criterion, optimizer=optimizer, mode='train')
        loss_epoch.append(loss)
        met_epoch.append(metrics)
    mean_loss = np.mean(loss_epoch)
    mean_metrics = np.mean(np.array(met_epoch), axis=0)
    return mean_loss, mean_metrics

def e_test(args, loader, encoder, classifier, criterion):
    encoder.eval()
    classifier.eval()
    loss_epoch = []
    met_epoch = []
    for _, (x, _, y) in enumerate(loader):
        loss, metrics = process_batch(args, x, x_aug=None, y=y, encoder=encoder, classifier=classifier, criterion=criterion, mode='eval')
        loss_epoch.append(loss)
        met_epoch.append(metrics)
    mean_loss = np.mean(loss_epoch)
    mean_metrics = np.mean(np.array(met_epoch), axis=0)
    return mean_loss, mean_metrics
