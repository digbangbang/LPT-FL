import torch
import torch
import numpy as np
import torch.nn as nn
import wandb

from args import parse_argument

args = parse_argument()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_epoch(model, loader):
    running_loss, samples = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
        running_loss = running_loss / samples
    return running_loss


def train_loss(model, loader):
    model.eval()
    running_loss, samples = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = nn.CrossEntropyLoss()(model(x), y)
        running_loss += loss.item() * y.shape[0]
        samples += y.shape[0]
    return running_loss / samples


def train_op(model, loader, optimizer, epochs,  quant_fn=None, lambda_fedprox=0.0, id=None):
    model.train()
    running_loss, samples = 0.0, 0
    weight_Q = quant_fn['weight_Q']
    grad_Q = quant_fn['grad_Q']

    if lambda_fedprox > 0.0:
        W0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    for ep in range(epochs):
        for it, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predict = model(x)
            loss = nn.CrossEntropyLoss()(predict, y)
            if lambda_fedprox > 0.0:
                loss += lambda_fedprox * torch.sum(
                    (flatten(W0).cuda() - flatten(dict(model.named_parameters())).cuda()) ** 2)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data = grad_Q(param.grad.data).data
            optimizer.step()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.data = weight_Q(p.data).data
            
            if id==args.client_id and args.test_client:
                wandb.log({'Client loss': loss,})

    return {"loss": running_loss / samples}


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = nn.CrossEntropyLoss()(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train_op_ma(model, loader, optimizer, epochs,  quant_fn=None, moving_weight=0.1):
    model.train()
    running_loss, samples = 0.0, 0
    weight_Q = quant_fn['weight_Q']
    grad_Q = quant_fn['grad_Q']

    grad_moving_avg = {}
    param_moving_avg = {}
    for name, param in model.named_parameters():
        grad_moving_avg[name] = torch.zeros_like(param)
        param_moving_avg[name] = torch.zeros_like(param)

    for ep in range(epochs):

        for it, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if it == 0:
                        grad_moving_avg[name] = param.grad.data
                    else:
                        grad_moving_avg[name] = moving_weight * grad_Q(grad_moving_avg[name]).data + (1 - moving_weight) * grad_Q(param.grad.data).data
                    param.grad.data = grad_Q(grad_moving_avg[name].data).data

            optimizer.step()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if it == 0:
                        param_moving_avg[name] = p.data
                    else:
                        param_moving_avg[name] = moving_weight * weight_Q(param_moving_avg[name]).data + (1 - moving_weight) * weight_Q(p.data).data
                    p.data = weight_Q(param_moving_avg[name].data).data

    return {"loss": running_loss / samples}


def eval_op(model, loader):
    model.train()
    samples, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.detach(), 1)
            loss = nn.CrossEntropyLoss()(y_, y).item()

            running_loss += loss * y.shape[0]
            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return {"accuracy": correct / samples, "loss": running_loss / samples}


def eval_op_ensemble(model, test_loader):
    model.eval()

    samples, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.detach(), 1)
            
            running_loss += nn.CrossEntropyLoss()(y_, y).item() * y_.shape[0]

            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    test_acc = correct / samples
    test_loss = running_loss / samples

    return {"test_accuracy": test_acc, "test_loss": test_loss}


def reduce_average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data.detach() for source in sources]), dim=0).clone()


def reduce_median(target, sources):
    for name in target:
        target[name].data = torch.median(torch.stack([source[name].detach() for source in sources]),
                                         dim=0).values.clone()


def reduce_weighted(target, sources, weights):
    for name in target:
        target[name].data = torch.sum(weights * torch.stack([source[name].detach() for source in sources], dim=-1),
                                      dim=-1).clone()


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def parse_dict(d, args):
    for key, value in d.items():
        if type(value) == dict:
            parse_dict(value, args)
        else:
            args.__dict__.setdefault(key, value)


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= alpha
        param1.data += param2.data * (1.0 - alpha)


def get_class_number(clients, n_class):
    client_class_num = np.zeros((len(clients), n_class))
    for client in clients:
        for x, bt_y in client.loader:
            for y in bt_y:
                client_class_num[client.id, y.item()] += 1

    return client_class_num


def generate_labels(number, class_num):
    labels = np.arange(number)
    proportions = class_num / class_num.sum()
    proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)


def get_batch_weight(labels, class_client_weight):
    bs = labels.size
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = class_client_weight[labels, :]
    return batch_weight

