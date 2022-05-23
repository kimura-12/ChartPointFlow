import argparse
import os

def add_args(parser):

    # point generator
    parser.add_argument('--h_dims_AF', type=str, default='128-128-128')
    parser.add_argument('--n_flow_AF', type=int, default=12)
    parser.add_argument('--nonlinearity', type=str, default='tanh')

    # prior flow
    parser.add_argument('--latent_dims', type=str, default='256')
    parser.add_argument("--latent_num_blocks", type=int, default=1)
    parser.add_argument("--layer_type", type=str, default="concatsquash")
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--bn_lag', type=float, default=0)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])


    # Chart Predictor architecture
    parser.add_argument('--cp_hdim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=12)
    parser.add_argument('--use_gumbel', type=eval, default=False)
    parser.add_argument('--mu', type=float, default=0.05)
    parser.add_argument('--lmd', type=float, default=1.)
    parser.add_argument('--tempe', type=float, default=1e-1)

    # training configuration
    parser.add_argument('--s_X_dim', type=int, default=128,
                        help='Dimension of the shape code')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=5000,
                        help='Step size for scheduler.')
    parser.add_argument('--gamma', type=float, default=0.25,
                        help='Learning rate decay ratio.')
    parser.add_argument('--stop_scheduler', type=int, default=15000,
                        help='When to freeze leraning rate.')
    parser.add_argument('--epochs', type=int, default=15000,
                        help='Number of epochs for training (default: 12000)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for training.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for genration and test.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed for initializing training. ')

    # data options
    parser.add_argument('--cates', type=str, nargs='+', default=["airplane"],
                        help="Categories to be trained")
    parser.add_argument('--data_dir', type=str, default="data/ShapeNetCore.v2.PC15k",
                        help="Path to the training data")
    parser.add_argument('--dataset_scale', type=float, default=1.,
                        help='Scale of the dataset (x,y,z * scale = real output, default=1).')
    parser.add_argument('--normalize_per_shape', action='store_true',
                        help='Whether to perform normalization per shape.')
    parser.add_argument('--normalize_std_per_axis', action='store_true',
                        help='Whether to perform normalization per axis.')
    parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (train)')
    parser.add_argument("--te_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (test)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading threads')

    # logging and saving frequency
    parser.add_argument('--save_dir', type=str, default="results/test")
    parser.add_argument('--vis_freq', type=int, default=400)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:5555', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # Evaluation options
    parser.add_argument('--num_sample_shapes', default=10, type=int,
                        help='Number of shapes to be sampled (for demo.py).')
    parser.add_argument('--num_sample_points', default=2048, type=int,
                        help='Number of points (per-shape) to be sampled (for demo.py).')
    parser.add_argument('--reconst_eval', type=eval, default=False,
                        help='select evaluation generation or reconstruction')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='ChartPointFlow Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    return args

