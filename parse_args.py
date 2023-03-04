import argparse
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(description='AdaGIn')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 1)')
    parser.add_argument('--source_dataset', type=str, default='dblpv7')
    parser.add_argument('--target_dataset', type=str, default='acmv9') 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_cly', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--aggregator_class', type=str, default='mean')
    parser.add_argument('--n_samples', type=str, default='10,10')
    parser.add_argument('--output_dims', type=str, default='256,64')
    parser.add_argument('--arch_cly', type=str, default="", help='node classifier architecture')
    parser.add_argument('--arch_disc', type=str, default="32-16", help='domain discriminator architecture')
    parser.add_argument('--is_social_net', action='store_true', help='whether to analysis the social networks')
    parser.add_argument('--is_blog', action='store_true', help='whether to analysis the blog networks, i.e., Blog1 and Blog2.')
    parser.add_argument('--dgi_param', type=float, default=1.0)
    parser.add_argument('--cdan_param', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    return args
