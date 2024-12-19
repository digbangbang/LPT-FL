import argparse


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, help='config path')
    parser.add_argument('--config_save_name', default=None, type=str, help='save the current config to this path')
    parser.add_argument("--project_name", default='quant FL', type=str)
    parser.add_argument("--data_path", default='/data/', type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--result_path", default=None, type=str)
    parser.add_argument("--acc_path", default=None, type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument('--dataset', type=str, default='fmnist', help='dataset')
    parser.add_argument('--ipc', type=int, default=15, help='image(s) per class')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for local data')
    parser.add_argument('--test_batch_size', type=int, default=256, help='batch size for test')
    parser.add_argument('--optimizer', default='Adam', help='dat  aset path')
    parser.add_argument('--classes', type=int, default=None, nargs='+', help='num of classes')
    parser.add_argument('--num_clients', type=int, default=80)
    parser.add_argument('--frac', type=float, default=0.4, help='clients participating rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--c_rounds', type=int, default=200, help='number of communication rounds')
    parser.add_argument('--alpha', type=float, default=0.04, help='alpha for dirichlet')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='ConvNet')
    parser.add_argument('--aggregation_mode', type=str, default='fedavg')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default=None, type=str, help='learning rate schedule')

    parser.add_argument('--use_quantization', action='store_true', default=False)
    parser.add_argument('--quantization_bits', type=int, default=8)
    parser.add_argument('--small_block', type=str, default='FC', choices=['Conv', 'FC'])
    parser.add_argument('--block_dim', type=str, default='B', choices=['B', 'BC'])
    parser.add_argument('--quant_type', type=str, default='stochastic', choices=['stochastic', 'nearest'])
    parser.add_argument('--m', type=int, default=6, help="floating point number mantissa")

    parser.add_argument('--moving_average', action='store_true', default=False, help="using moving average after aggregation")
    parser.add_argument('--moving_weight', type=float, default=0., help="coefficient for moving statistics in BN")
    parser.add_argument('--ma_start', type=int, default=1, help="start moving average after .. communication")

    ## FedProx 
    parser.add_argument('--lambda_fedprox', default=0.001, type=float, help='lambda of fedprox')

    ## FedGen
    parser.add_argument("--ensemble_lr", type=float, default=3e-4)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--generative_alpha", type=float, default=0.1)
    parser.add_argument("--generative_beta", type=float, default=2.5)
    parser.add_argument("--ensemble_alpha", type=float, default=1)
    parser.add_argument("--ensemble_beta", type=float, default=0)
    parser.add_argument("--ensemble_eta", type=float, default=0)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--use_embedding", type=int, default=0)
    parser.add_argument("--coef_decay", type=float, default=0.98)
    parser.add_argument("--coef_decay_epoch", type=int, default=1)
    parser.add_argument("--ensemble_epoch", type=int, default=50)
    parser.add_argument("--train_generator_epoch", type=int, default=5)
    parser.add_argument("--min_samples_per_label", type=int, default=1)

    ## FedTGP
    parser.add_argument("--g_lr", type=float, default=1e-2)
    parser.add_argument("--s_lr", type=float, default=1e-4)
    parser.add_argument("--tgp_iteration", type=int, default=10)
    parser.add_argument("--inner_round_g", type=int, default=1)
    parser.add_argument("--inner_round_d", type=int, default=5)

    ## Test client
    parser.add_argument('--test_client', action='store_true', default=False, help="test client")
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument('--global_test', action='store_true', default=False, help="test global test")

    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help='whether useing mixed precision local training')
    parser.add_argument('--weight_quantization_bits', type=int, default=8)
    parser.add_argument('--activation_quantization_bits', type=int, default=8)
    parser.add_argument('--grad_quantization_bits', type=int, default=8)

    args = parser.parse_args()
    return args

