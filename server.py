import random
import torch
import models as model_utils
import torch.nn.functional as F

from utils import *
from client import Device
from copy import deepcopy


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server(Device):
    def __init__(self, model_name, loader, num_classes=10, dataset='cifar10', moving_weight=0.1, quant=None, mode=None):
        super().__init__(loader)
        print(f"dataset server {dataset}")
        if model_name == 'Transformer':
            self.model = partial(model_utils.get_model('Transformer')[0], quant=quant)().to(device)
        else:
            self.model = partial(model_utils.get_model(model_name)[0], num_classes=num_classes, net_norm='batchnorm' if mode=='fedbn' else 'instancenorm', dataset=dataset, quant=quant)().to(device)

        self.parameter_dict = {key: value for key, value in self.model.named_parameters()}
        self.moving_weight = moving_weight
        self.num_classes = num_classes
        self.dataset = dataset

        if mode=='fedgen':
            from args import parse_argument
            self.args = parse_argument()
            self.generator = Generator(self).to(device)
            self.generator_optimizer = torch.optim.Adam(
                self.generator.parameters(), args.ensemble_lr
            )
            self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.generator_optimizer, gamma=0.98
            )
            self.unique_labels = range(self.num_classes)
            self.teacher_model = deepcopy(self.model)
        if mode=='fedtgp':
            from args import parse_argument
            self.args = parse_argument()
            self.nz = 128
            self.nc = 3 if dataset in ['cifar10', 'cinic10', 'cifar100'] else 1
            self.img_sz = 32 if dataset in ['cifar10', 'cinic10', 'cifar100'] else 28
            self.generator = partial(model_utils.get_model('Generator'), nz=self.nz, nc=self.nc, img_size=self.img_sz, n_cls=num_classes)().to(device)

            lr_schedule = getattr(torch.optim.lr_scheduler, self.args.lr_schedule)
            g_lr_schedule_fn = lambda x: lr_schedule(x, T_max=self.args.c_rounds, eta_min=1e-4)
            s_lr_schedule_fn = lambda x: lr_schedule(x, T_max=self.args.c_rounds, eta_min=1e-5)

            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), self.args.g_lr)
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.args.s_lr)
            self.g_lr_schedule = g_lr_schedule_fn(self.generator_optimizer)
            self.s_lr_schedule = s_lr_schedule_fn(self.model_optimizer)

    def evaluate_ensemble(self):
        return eval_op_ensemble(self.model, self.loader)

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients) * frac))

    def fedavg(self, clients, bn=True):
        if bn:
            for name, param in self.model.named_parameters():
                param.data = torch.mean(
                    torch.stack([dict(c.model.named_parameters())[name].data.detach() for c in clients]),
                    dim=0)
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    continue
                if hasattr(module, 'weight'):
                    module.weight.data = torch.mean(
                        torch.stack([c.model.state_dict()[f'{name}.weight'] for c in clients]),
                        dim=0
                    )
                if hasattr(module, 'bias'):
                    module.bias.data = torch.mean(
                        torch.stack([c.model.state_dict()[f'{name}.bias'] for c in clients]),
                        dim=0
                    )

    def abavg(self, clients):
        acc = torch.zeros([len(clients)], device="cuda")
        for j, (x, true_y) in enumerate(self.loader):
            if self.dataset == 'cinic10' and j == 20:
                break
            x = x.to(device)
            true_y = true_y.to(device)
            for i, client in enumerate(clients):
                y_ = client.predict_logit(x)
                _, predicted = torch.max(y_.detach(), 1)
                acc[i] += (predicted == true_y).sum().item()

        self.weights = acc / acc.sum()
        reduce_weighted(target=self.parameter_dict,
                        sources=[client.W for client in clients],
                        weights=self.weights)

    def fedgen(self, label_counts_cache, client_params_cache, weight_cache):
        self.train_generator(client_params_cache, label_counts_cache)
        weights = torch.tensor(weight_cache, device=device) / sum(weight_cache)

        client_params = []
        for client_state_dict in client_params_cache:
            client_params.append([value for name, value in client_state_dict.items()])

        aggregated_params = [
            torch.sum(weights * torch.stack(layer_params, dim=-1), dim=-1)
            for layer_params in zip(*client_params)
        ]
        for old_param, new_param in zip(
            self.model.parameters(), aggregated_params
        ):
            old_param.data.copy_(new_param)

    def fedtgp(self, clients, client_class_num):
        self.fedavg(clients)

        for client in clients:
            client.model.eval()
            client.model.to(device)
        self.generator.to(device)

        num_clients, num_classes = client_class_num.shape

        class_num = np.sum(client_class_num, axis=0)
        class_client_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
        class_client_weight = class_client_weight.transpose()
        labels_all = generate_labels(self.args.tgp_iteration * self.args.batch_size, class_num)

        for i in range(self.args.tgp_iteration):
            labels = labels_all[i * self.args.batch_size: (i+1) * self.args.batch_size]
            batch_weight = torch.Tensor(get_batch_weight(labels, class_client_weight)).cuda()
            onehot = np.zeros((self.args.batch_size, num_classes))
            onehot[np.arange(self.args.batch_size), labels] = 1
            y_onehot = torch.Tensor(onehot).cuda()
            z = torch.randn((self.args.batch_size, self.nz, 1, 1)).cuda()

            self.model.eval()
            self.generator.train()

            for _ in range(self.args.inner_round_g):
                for i in range(num_clients):
                    self.generator_optimizer.zero_grad()
                    t_model = clients[i].model
                    self.compute_backward_flow_G_dis(z, y_onehot, labels, self.generator, self.model, t_model, batch_weight[:, i], num_clients)
                    self.generator_optimizer.step()

            self.model.train()
            self.generator.eval()
            for _ in range(self.args.inner_round_d):
                self.model_optimizer.zero_grad()
                fake = self.generator(z, y_onehot).detach()
                s_logit = self.model(fake)
                t_logit_merge = 0
                for i in range(num_clients):
                    t_model = clients[i].model
                    t_logit = t_model(fake).detach()
                    t_logit_merge += F.softmax(t_logit, dim=1) * batch_weight[:, i][:, np.newaxis].repeat(1, num_classes)
                loss_D = torch.mean(-F.log_softmax(s_logit, dim=1) * t_logit_merge)
                loss_D.backward()
                self.model_optimizer.step()

        self.g_lr_schedule.step()
        self.s_lr_schedule.step()

    def train_generator(self, client_params_cache=None, label_counts_cache=None):
        self.generator.train()
        self.teacher_model.eval()
        self.model.eval()
        label_weights, qualified_labels = self.get_label_weights(label_counts_cache)
        for _ in range(self.args.train_generator_epoch):
            y_npy = np.random.choice(qualified_labels, self.args.batch_size)
            y_tensor = torch.tensor(y_npy, dtype=torch.long, device=device)

            generator_output, eps = self.generator(y_tensor)

            diversity_loss = self.generator.diversity_loss(eps, generator_output)

            teacher_loss = 0
            teacher_logit = 0

            for i, model_state_dict in enumerate(client_params_cache):
                self.teacher_model.load_state_dict(model_state_dict, strict=True)
                weight = label_weights[y_npy][:, i].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, len(self.unique_labels)))
                logits = self.model.classifier(generator_output)
                teacher_loss += torch.mean(
                    self.generator.ce_loss(logits, y_tensor)
                    * torch.tensor(weight, dtype=torch.float, device=device)
                )
                teacher_logit += logits * torch.tensor(
                    expand_weight, dtype=torch.float, device=device
                )

            student_logits = self.model.classifier(generator_output)
            student_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1), F.softmax(teacher_logit, dim=1)
            )
            loss = (
                    self.args.ensemble_alpha * teacher_loss
                    - self.args.ensemble_beta * student_loss
                    + self.args.ensemble_eta * diversity_loss
            )
            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

        self.generator_lr_scheduler.step()

    def get_label_weights(self, label_counts_cache):
        label_weights = []
        qualified_labels = []
        for i, label_counts in enumerate(zip(*label_counts_cache)):
            count_sum = sum(label_counts)
            label_weights.append(np.array(label_counts) / count_sum)
            if (
                count_sum / len(label_counts_cache)
                > self.args.min_samples_per_label
            ):
                qualified_labels.append(i)
        label_weights = np.array(label_weights).reshape((len(self.unique_labels)), -1)
        return label_weights, qualified_labels

    def compute_backward_flow_G_dis(self, z, y_onehot, labels, generator, s_model, t_model, weight, num_clients):
        lambda_cls = 1.0
        lambda_dis = 1.0

        cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        diversity_criterion = DiversityLoss(metric='l1')

        y = torch.Tensor(labels).long().to(device)

        fake = generator(z, y_onehot)
        t_logit = t_model(fake)
        s_logit = s_model(fake)
        loss_md = - torch.mean(torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)
        loss_cls = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())
        loss_ap = diversity_criterion(z.view(z.shape[0], -1), fake)
        loss = loss_md + lambda_cls * loss_cls + lambda_dis * loss_ap / num_clients

        loss.backward()
    
    def fedbn_test(self, clients):
        correct_count = 0
        total_count = 0
        for j, (x, true_y) in enumerate(self.loader):
            if self.dataset == 'cinic10' and j == 20:
                break
            x = x.to(device)
            true_y = true_y.to(device)
            predict_y = []
            for i, client in enumerate(clients):
                y_ = client.predict_logit(x)
                _, predicted = torch.max(y_.detach(), 1)
                predict_y.append(predicted)
            predict_tensor = torch.stack(predict_y).t()
            voted_predictions = torch.mode(predict_tensor)[0]

            correct_count += (voted_predictions == true_y).sum().item()
            total_count += true_y.size(0)
        accuracy = correct_count / total_count
        return accuracy

            
class Generator(nn.Module):
    def __init__(self, server: Server) -> None:
        super().__init__()
        data_iter = iter(server.loader)
        x0, y0 = next(data_iter)
        x = torch.zeros(x0.shape, device=device)

        self.device = device
        self.use_embedding = server.args.use_embedding
        self.latent_dim = server.model.base(x).shape[-1]
        self.hidden_dim = server.args.hidden_dim
        self.noise_dim = server.args.noise_dim
        self.class_num = len(range(server.num_classes))

        if server.args.use_embedding:
            self.embedding = nn.Embedding(self.class_num, server.args.noise_dim)
        input_dim = (
            self.noise_dim * 2
            if server.args.use_embedding
            else self.noise_dim + self.class_num
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.diversity_loss = DiversityLoss(metric="l1")
        self.dist_loss = nn.MSELoss()

    def forward(self, targets):
        batchsize = targets.shape[0]
        eps = torch.randn((batchsize, self.noise_dim), device=self.device)
        if self.use_embedding:
            y = self.embedding(targets)
        else:
            y = torch.zeros((batchsize, self.class_num), device=self.device)
            y.scatter_(1, targets.reshape(-1, 1), 1)
        z = torch.cat([eps, y], dim=1)
        z = self.mlp(z)
        z = self.latent_layer(z)
        return z, eps


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=2)
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=2)
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


    
