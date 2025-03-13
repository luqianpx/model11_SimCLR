import os
import torch
import support_based as spb
from support_based import com_mul_str
from simclr.simclr import SimCLR_encoder
from simclr.modules import NT_Xent, MLP_Classifier
from support_model import load_optimizer, save_model, load_model
from support_dataset import ECG_dataset
from model_optimization import train, finetune_train, e_test

def initialize_device_and_model(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    # Initialize the model
    model = SimCLR_encoder(args).to(args.device)
    return model

def save_model_if_best(model, sa_fo, epoch_loss, lowest_loss, loss_li, args):
    if epoch_loss < lowest_loss:
        save_model(sa_fo, 'pretrain-main', model)
        lowest_loss = epoch_loss
    loss_li.append(epoch_loss)
    spb.save_res(sa_fo, 'pretrain-res', [loss_li, args])
    return lowest_loss, loss_li

def pretrain(args):
    DA = ECG_dataset(args)

    sa_fo = './save/' + com_mul_str(args) + '/'
    os.makedirs(sa_fo, exist_ok=True)

    model = initialize_device_and_model(args)
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature)

    args.current_epoch = 0
    lowest_loss = float('inf')
    loss_li = []

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, DA, model, criterion, optimizer)
        scheduler.step()
        print(f"Epoch {epoch}: Loss {loss_epoch}")
        
        lowest_loss, loss_li = save_model_if_best(model, sa_fo, loss_epoch, lowest_loss, loss_li, args)
        args.current_epoch += 1

    return

def finetune(args):
    model = initialize_device_and_model(args)
    sa_fo = './save/' + args.sa_folder + '/'

    if args.downstream_tr_type != 'nopretrain':
        load_model(sa_fo, 'pretrain-main', model)

    args.dataset_type = 'finetune'
    tr_DA = ECG_dataset(args)
    args.dataset_type = 'test'
    te_DA = ECG_dataset(args)

    classifier = MLP_Classifier(model.n_features, args.n_class).to(args.device)

    if args.downstream_tr_type == "pretrain_only_classifier":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
    else:
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': classifier.parameters()}], lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    highest_F1 = 0.1
    los_met_li = [[], [], args]

    for epoch in range(args.logistic_epochs):
        # Train
        mean_loss_train, mean_met_train = finetune_train(args, tr_DA, model, classifier, criterion, optimizer)
        print(f"Epoch {epoch}: Train Loss {mean_loss_train}, Metrics {mean_met_train}")
        los_met_li[0].append([mean_loss_train, mean_met_train])

        # Validation
        mean_loss_val, mean_met_val = e_test(args, te_DA, model, classifier, criterion)
        print(f"Epoch {epoch}: Validation Loss {mean_loss_val}, Metrics {mean_met_val}")
        los_met_li[1].append([mean_loss_val, mean_met_val])

        # Save model if the F1 score is the highest
        if mean_met_val[3] > highest_F1:
            save_model(sa_fo, f'finetune-{args.downstream_tr_type}-main', model)
            save_model(sa_fo, f'finetune-{args.downstream_tr_type}-classifier', classifier)
            highest_F1 = mean_met_val[3]

        spb.save_res(sa_fo, f'finetune-{args.downstream_tr_type}-res', los_met_li)

    return
