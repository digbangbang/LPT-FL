import os
import wandb
import json
import torch
import data
import random

from easydict import EasyDict
from utils import *
from args import parse_argument
from server import Server
from client import Client
from tqdm import tqdm
from quantizer import quantize_block, BlockQuantizer


def run():
    args = parse_argument()
    if args.config is not None:
        with open(args.config, 'r') as f:
            arg_dict = EasyDict(json.load(f))
            parse_dict(arg_dict, args)
    
    if args.result_path is not None and not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.use_quantization and args.use_mixed_precision:
        run_name = '_'.join(
            [str(args.moving_weight), args.dataset, args.model_name, str(args.alpha), str(args.weight_quantization_bits), str(args.activation_quantization_bits), str(args.grad_quantization_bits), args.aggregation_mode])
    elif args.use_quantization:
        run_name = '_'.join(
            [str(args.moving_weight), args.dataset, args.model_name, str(args.alpha), str(args.quantization_bits), args.aggregation_mode])
    else:
        run_name = '_'.join([str(args.moving_weight), args.dataset, args.model_name, str(args.alpha), args.aggregation_mode])

    if args.grad_clip:
        run_name += f'_gc_{args.clip_to}'

    if args.test_client:
        run_name += f'_{args.client_id}'

    wandb.init(project=args.project_name, name=run_name, config=args, entity='XXXX') # input ur own entity

    args.num_classes = \
        {"mnist": 10, "fmnist": 10, "cifar10": 10, "cinic10": 10, "cifar100": 100, "nlp": 4, 'news20': 20}[
            args.dataset]
    args.channel = {"cifar10": 3, "cinic10": 3, "cifar100": 3, "mnist": 1, "fmnist": 1}[args.dataset]
    args.imsize = {"cifar10": (32, 32),
                "cinic10": (32, 32),
                "cifar100": (32, 32),
                "mnist": (28, 28),
                "fmnist": (28, 28),
                }[args.dataset]

    if args.config_save_name is not None:
        with open("./configs/" + args.config_save_name + '.json', 'wt') as f:
            json.dump(vars(args), f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lr_schedule = getattr(torch.optim.lr_scheduler, args.lr_schedule) if args.lr_schedule is not None else None

    optimizer = getattr(torch.optim, args.optimizer)
    optimizer_fn = lambda x: optimizer(x, lr=args.lr)

    schedule_fn = lambda x: lr_schedule(x, T_max=args.c_rounds, eta_min=1e-5) if lr_schedule is not None else None

    train_data_all, test_data = data.get_data(args.dataset, args.data_path)
    client_loaders, test_loader = data.get_loaders(train_data_all, test_data, n_clients=args.num_clients,
                                                    alpha=args.alpha, batch_size=args.batch_size,
                                                    test_batch_size=args.test_batch_size, n_data=None,
                                                    num_workers=4, seed=args.seed)

    if args.use_quantization and args.use_mixed_precision:
        weight_quantizer = lambda x: quantize_block(
            x, args.weight_quantization_bits, -1, args.quant_type, args.small_block, args.block_dim)
        grad_quantizer = lambda x: quantize_block(
            x, args.grad_quantization_bits, -1, args.quant_type, args.small_block, args.block_dim)

        quant_model = lambda : BlockQuantizer(args.activation_quantization_bits, args.activation_quantization_bits, args.quant_type,
                                       args.small_block, args.block_dim)
        quant_server = lambda : BlockQuantizer(-1, -1, args.quant_type,
                                       args.small_block, args.block_dim)
    elif args.use_quantization:
        weight_quantizer = lambda x: quantize_block(
            x, args.quantization_bits, -1, args.quant_type, args.small_block, args.block_dim)
        grad_quantizer = lambda x: quantize_block(
            x, args.quantization_bits, -1, args.quant_type, args.small_block, args.block_dim)

        quant_model = lambda : BlockQuantizer(args.quantization_bits, args.quantization_bits, args.quant_type,
                                       args.small_block, args.block_dim)
        quant_server = lambda : BlockQuantizer(-1, -1, args.quant_type,
                                       args.small_block, args.block_dim)
    else:
        weight_quantizer = lambda x: quantize_block(
            x, -1, -1, args.quant_type, args.small_block, args.block_dim)
        grad_quantizer = lambda x: quantize_block(
            x, -1, -1, args.quant_type, args.small_block, args.block_dim)

        quant_model = lambda: BlockQuantizer(-1, -1, args.quant_type,
                                             args.small_block, args.block_dim)
        quant_server = lambda: BlockQuantizer(-1, -1, args.quant_type,
                                                      args.small_block, args.block_dim)

    quantizer = {'weight_Q' : weight_quantizer, 'grad_Q' : grad_quantizer}

    server = Server(args.model_name, test_loader, num_classes=args.num_classes, dataset=args.dataset, moving_weight=args.moving_weight,
                    quant=quant_server, mode=args.aggregation_mode)

    model = Server(args.model_name, test_loader, num_classes=args.num_classes, dataset=args.dataset, moving_weight=args.moving_weight,
                   quant=quant_model, mode=args.aggregation_mode)

    clients = [
        Client(args.model_name, optimizer_fn=optimizer_fn, loader=loader, idnum=i, num_classes=args.num_classes, dataset=args.dataset,
               lr_schedule=schedule_fn, quant=quant_model, mode=args.aggregation_mode) for
        i, loader in enumerate(client_loaders)]

    final_test_acc_list = []
    if args.aggregation_mode == 'fedtgp':
        client_class_num = get_class_number(clients, args.num_classes)

    for i in tqdm(range(1, args.c_rounds + 1)):
        selected_id = []
        for client in clients:
            client.synchronize_with_server(server, bn=True if args.aggregation_mode != 'fedbn' else False)
        client_loss = 0.
        participating_clients = server.select_clients(clients, args.frac)
        if args.aggregation_mode == 'fedgen':
            client_params_cache = []
            weight_cache = []
            label_counts_cache = []
        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.epochs, quant_fn=quantizer, lambda_fedprox=args.lambda_fedprox if args.aggregation_mode == 'fedprox' else 0.0, current_global_epoch=i, generator=server.generator if args.aggregation_mode == 'fedgen' else None,
                                                       regularization=i > 0)
            selected_id.append(client.id)
            if args.lr_schedule is not None:
                client.lr_schedule.step()

            client_loss += train_stats['loss']
            if args.aggregation_mode == 'fedgen':
                label_counts_cache.append(train_stats['label_counts'])
                client_params_cache.append(train_stats['delta'])
                weight_cache.append(train_stats['weight'])

        client_train_loss = client_loss / len(participating_clients)

        if args.test_client and args.client_id not in selected_id:
            _ = clients[args.client_id].compute_weight_update(epochs=args.epochs, quant_fn=quantizer, lambda_fedprox=0.0, c_global=None, current_global_epoch=0, generator=None, regularization=0)

        if args.aggregation_mode == 'fedavg':
            model.fedavg(participating_clients)
        elif args.aggregation_mode == 'abavg':
            model.abavg(participating_clients)
        elif args.aggregation_mode == 'fedprox':
            model.fedavg(participating_clients)
        elif args.aggregation_mode == 'fedgen':
            model.fedgen(label_counts_cache, client_params_cache, weight_cache)
        elif args.aggregation_mode == 'fedtgp':
            model.fedtgp(participating_clients, client_class_num[selected_id])
        elif args.aggregation_mode == 'fedbn':
            model.fedavg(participating_clients, bn=False)
            fedbn_acc = model.fedbn_test(clients)
        else:
            raise ValueError('{} is not set aggregation mode'.format(args.aggregation_mode))

        if args.moving_average and i>=args.ma_start:
            moving_average(server.model, model.model, args.moving_weight if i> args.ma_start else 0)
        elif not args.moving_average:
            server = model

        if args.moving_average and i<args.ma_start:
            eval_stats = model.evaluate_ensemble()
        else:
            eval_stats = server.evaluate_ensemble()

        test_acc, test_loss = eval_stats['test_accuracy'], eval_stats['test_loss']
        wandb.log({
            'Client loss': 0.,
            'train_loss': client_train_loss,
            'global_loss' if args.global_test else 'test_loss': test_loss,
            'test_accuracy': test_acc if args.aggregation_mode != 'fedbn' else fedbn_acc,
        })

        if i > args.c_rounds - 5:
            final_test_acc_list.append(test_acc)

    acc_arr = np.array(final_test_acc_list)
    mean = np.mean(acc_arr)
    var = np.var(acc_arr)
    std = np.std(acc_arr)
    wandb.log({'mean_final_test_acc_last_5c': mean, 'var_final_test_acc_last_5c': var,
               'std_final_test_acc_last_5c': std})


if __name__ == "__main__":
    run()
