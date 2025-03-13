import argparse
import os

def parse(args=None, test=None):
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--n_class', dest='n_class', type=int, default=6)
    parser.add_argument('--n_channel', dest='img_size', type=int, default=12)
    parser.add_argument('--n_length', dest='n_length', type=int, default=5000)
    parser.add_argument('--whe_mix_lead', dest='whe_mix_lead', type=str, default='nomix', choices=['mix', 'nomix'])

    # Model options
    parser.add_argument('--model_type', dest='model_type', type=str, default='resnet1d', choices=['resnet1d', 'cnntransf'])
    parser.add_argument('--projection_dim', dest='projection_dim', type=int, default=64)

    # ResNet1D specific options
    parser.add_argument('--base_filters', dest='base_filters', type=int, default=32)
    parser.add_argument('--kernel_size', dest='kernel_size', type=int, default=10)
    parser.add_argument('--stride', dest='stride', type=int, default=2)
    parser.add_argument('--groups', dest='groups', type=int, default=2)
    parser.add_argument('--n_block', dest='n_block', type=int, default=20)
    parser.add_argument('--downsample_gap', dest='downsample_gap', type=int, default=4)
    parser.add_argument('--increasefilter_gap', dest='increasefilter_gap', type=int, default=4)

    # CNN Transformer specific options
    parser.add_argument('--fft', dest='fft', type=int, default=200)
    parser.add_argument('--steps', dest='steps', type=int, default=20)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)
    parser.add_argument('--nhead', dest='nhead', type=int, default=8)
    parser.add_argument('--emb_size', dest='emb_size', type=int, default=256)
    parser.add_argument('--n_segments', dest='n_segments', type=int, default=10)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=20)

    # Loss and optimizer options
    parser.add_argument('--optimizer', dest='optimizer', type=str, default="AdamW")
    parser.add_argument('--lr', dest='lr', type=float, default=1.5e-5)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.05)
    parser.add_argument('--temperature', dest='temperature', type=float, default=0.5)
    parser.add_argument('--warmup_epoch', dest='warmup_epoch', type=int, default=5)

    # Save model folder
    parser.add_argument('--sa_folder', dest='sa_folder', type=str, default="resnet1d_nomix_1500_400_2500_20_OmNi")

    # Fine-tuning options
    parser.add_argument('--dataset_type', dest='dataset_type', type=str, default='pretrain', choices=['pretrain', 'finetune', 'test'])
    parser.add_argument('--downstream_tr_type', dest='downstream_tr_type', type=str, default='nopretrain', choices=['nopretrain', 'pretrain_full', 'pretrain_only_classifier'])
    parser.add_argument('--labelled_ratio', dest='labelled_ratio', type=float, default=0.1)

    # Define environment-specific options for batch size and epochs
    is_hpc = os.path.isdir('/share/home/hulianting/Project/Project20_ECG_foundation_model/') or test == 'ser'

    if is_hpc:
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
        parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0)
        parser.add_argument('--epochs', dest='epochs', type=int, default=1500)
        parser.add_argument('--running_env', dest='running_env', type=str, default='HPC')

        # Logistic regression options
        parser.add_argument('--logistic_batch_size', dest='logistic_batch_size', type=int, default=256)
        parser.add_argument('--logistic_epochs', dest='logistic_epochs', type=int, default=100)

    else:
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
        parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0)
        parser.add_argument('--epochs', dest='epochs', type=int, default=2)
        parser.add_argument('--running_env', dest='running_env', type=str, default='local')

        # Logistic regression options
        parser.add_argument('--logistic_batch_size', dest='logistic_batch_size', type=int, default=30)
        parser.add_argument('--logistic_epochs', dest='logistic_epochs', type=int, default=500)

    return parser.parse_args(args)
