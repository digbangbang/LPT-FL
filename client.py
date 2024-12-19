import copy
import torch
import models as model_utils
import torch.nn.functional as F

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Device(object):
    def __init__(self, loader):
        self.loader = loader

    def evaluate(self, loader=None):
        return eval_op(self.model, self.loader if not loader else loader)

    def save_model(self, path=None, name=None, verbose=True):
        if name:
            torch.save(self.model.state_dict(), path + name)
            if verbose: print("Saved model to", path + name)

    def load_model(self, path=None, name=None, verbose=True):
        if name:
            self.model.load_state_dict(torch.load(path + name))
            if verbose: print("Loaded model from", path + name)


class Client(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10', lr_schedule=None,
                 quant=None, mode=None):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes, net_norm='batchnorm' if mode=='fedbn' else 'instancenorm', dataset=dataset,
                                quant=quant)().to(device)
        self.unique_target = [i for i in range(num_classes)]

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer = optimizer_fn(self.model.parameters())
        if lr_schedule is not None:
            self.lr_schedule = lr_schedule(self.optimizer)

        self.mode = mode
        self.dataset = dataset

    def synchronize_with_server(self, server, bn=True):
        if bn:
            server_state = server.model.state_dict()
            self.model.load_state_dict(server_state, strict=True)

            self.origin = copy.deepcopy(self.model)
        else:
            bn_name_list = []
            server_state = server.model.state_dict()
            for name, module in self.model.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    bn_name_list.append(name)
            def filter_bn_params(state_dict, bn_name_list):
                filtered_state_dict = {}
                for key, param in state_dict.items():
                    if not any(bn_name in key for bn_name in bn_name_list):
                        filtered_state_dict[key] = param
                return filtered_state_dict

            filtered_server_state = filter_bn_params(server_state, bn_name_list)

            self.model.load_state_dict(filtered_server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None, quant_fn=None, lambda_fedprox=0.0, current_global_epoch=None, generator=None, regularization=0):
        if self.mode == 'fedgen':
            from args import parse_argument
            self.args = parse_argument()
            self.current_global_epoch = current_global_epoch

            all_targets = []
            target_count = {target: 0 for target in self.unique_target}

            for batch_idx, (inputs, targets) in enumerate(self.loader):
                all_targets.extend(targets.tolist())
                for i in range(targets.size(0)):
                    target_count[targets[i].item()] += 1
            self.available_labels = torch.unique(torch.tensor(all_targets)).tolist()
            target_list = [target_count[target] if target in target_count else 1 for target in self.unique_target]

            weight_Q = quant_fn['weight_Q']
            grad_Q = quant_fn['grad_Q']
            self.model.train()
            generator.train()
            running_loss, samples = 0.0, 0
            for it in range(epochs):
                for x, y in self.loader:
                    x, y = x.to(device), y.to(device)
                    logits = self.model(x)
                    loss = nn.CrossEntropyLoss()(logits, y)

                    if regularization:
                        alpha = self.exp_coef_scheduler(self.args.generative_alpha)
                        beta = self.exp_coef_scheduler(self.args.generative_beta)
                        generator_output, _ = generator(y)
                        logits_gen = self.model.classifier(generator_output).detach()

                        latent_loss = beta * F.kl_div(
                            F.log_softmax(logits, dim=1),
                            F.softmax(logits_gen, dim=1),
                            reduction="batchmean",
                        )

                        sampled_y = torch.tensor(
                            np.random.choice(
                                self.available_labels, self.args.gen_batch_size
                            ),
                            dtype=torch.long,
                            device=device,
                        )
                        generator_output, _ = generator(sampled_y)
                        logits = self.model.classifier(generator_output)
                        teacher_loss = alpha * nn.CrossEntropyLoss()(logits, sampled_y)

                        gen_ratio = self.args.gen_batch_size / self.args.batch_size

                        loss += gen_ratio * teacher_loss + latent_loss

                    running_loss += loss.item() * y.shape[0]
                    samples += y.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            param.grad.data = grad_Q(param.grad.data).data

                    self.optimizer.step()
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            param.data = weight_Q(param.data).data


            delta = self.model.state_dict()

            return {"loss": running_loss / samples, "delta": delta, "weight": len(self.loader), "label_counts": target_list}

        else:
            train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs,
                                   quant_fn=quant_fn, lambda_fedprox=lambda_fedprox, id=self.id)
        return train_stats

    def compute_weight_update_ma(self, epochs=1, loader=None, quant_fn=None, moving_weight=0.1):
        train_stats = train_op_ma(self.model, self.loader if not loader else loader, self.optimizer, epochs,
                               quant_fn=quant_fn, moving_weight=moving_weight)
        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.eval()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def exp_coef_scheduler(self, init_coef):
        return max(
            1e-4,
            init_coef
            * (
                self.args.coef_decay
                ** (self.current_global_epoch // self.args.coef_decay_epoch)
            ),
        )
