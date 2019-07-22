import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm


class Net_FeatureExtractore(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x


class Net_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from abc import ABCMeta


class BaseMCD(nn.Module, metaclass=ABCMeta):
    pass


class Net(BaseMCD):
    def __init__(self):
        super(Net, self).__init__()
        self.generator = Net_FeatureExtractore()
        self.classifier_f1 = Net_Classifier()
        self.classifier_f2 = Net_Classifier()

    def forward(self, x):
        x = self.generator(x)
        x = x.view(-1, 4 * 4 * 50)
        x_f1 = self.classifier_f1(x)
        x_f2 = self.classifier_f2(x)
        return x_f1, x_f2

    def parameters(self, recurse=True):
        gen_params = self.generator.parameters(recurse=recurse)

        cls_params = []
        cls_params.extend(list(self.classifier_f1.parameters(recurse=recurse)))
        cls_params.extend(list(self.classifier_f2.parameters(recurse=recurse)))

        return gen_params, cls_params



from abc import ABCMeta


class Config:
    class BaseChildConfig(metaclass=ABCMeta):
        pass

    class Optim(BaseChildConfig):
        name = "sgd"
        lr_g = 0.00001
        lr_c = 0.001
        momentum = 0.5

    def __init__(self):
        self.seed = 1
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 1
        self.log_interval = 10
        self.save_model = ""
        self.optim = self.Optim()


class Device:
    def __init__(self, N_GPUs: int):
        self.N_GPUs = N_GPUs
        self.is_cuda = False
        if not torch.cuda.is_available():
            dev = "cpu"
        elif self.N_GPUs <= 0:
            dev = "cpu"
        else:
            dev = "cuda"
            self.is_cuda = True

        self.__device = torch.device(dev)

    def get(self):
        return self.__device


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class TwoDomainDataLoader:
    def __init__(self, device: Device, batch_size: int, test_batch_size: int):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device.is_cuda else {}

        DATA_ROOT = "/raid/pytorch"
        source = datasets.MNIST(
            str(DATA_ROOT), train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        target = datasets.MNIST(
            str(DATA_ROOT), train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        train_dataset = ConcatDataset(source, target)
        val_dataset = ConcatDataset(source, target)
        self.train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)
        self.val = torch.utils.data.DataLoader(
            val_dataset, batch_size=test_batch_size,
            shuffle=True, **kwargs)

    @property
    def train_len(self):
        return len(self.train.dataset)

    @property
    def val_len(self):
        return len(self.val.dataset)


from abc import ABCMeta


class BaseGenClsOptimizer(metaclass=ABCMeta):
    generator = None
    classifier = None


class Optimizer(BaseGenClsOptimizer):
    def __init__(self, model: BaseMCD, cfg: Config.Optim):
        model_params = model.parameters()
        self.generator = optim.SGD(
            model_params[0],
            lr=cfg.lr_g, momentum=cfg.momentum
        )
        self.classifier = optim.SGD(
            model_params[1],
            lr=cfg.lr_c, momentum=cfg.momentum
        )

    def zero_grad(self):
        self.generator.zero_grad()
        self.classifier.zero_grad()


class OptunaOptimizer(BaseGenClsOptimizer):
    def __init__(self, model: BaseMCD, cfg: Config, trial):
        lr_c = trial.suggest_loguniform("lr_c", 1e-10, 1e-3)
        lr_d = trial.suggest_loguniform("lr_d", 1e-10, 1e-3)
        model_params = model.parameters()
        self.generator = optim.SGD(
            model_params[0],
            lr=cfg.optim.lr_g, momentum=cfg.optim.momentum
        )
        self.classifier = optim.SGD(
            model_params[1],
            lr=cfg.optim.lr_c, momentum=cfg.optim.momentum
        )

    def zero_grad(self):
        self.generator.zero_grad()
        self.classifier.zero_grad()


class Diff2d(nn.Module):
    def __init__(self, weight=None):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1)))


class Criterion(nn.Module):
    CEL = "cross_entropy"

    ### for discrepancy
    diff = "diff"
    mysymkl = "mysymkl"
    symkl = 'symkl'

    def __init__(self, name=CEL, reduction="mean"):
        super().__init__()
        self.name = name
        if name == self.CEL:
            # self.loss_func = nn.CrossEntropyLoss()
            self.loss_func = nn.NLLLoss(reduction=reduction)
        elif name == self.diff:
            self.loss_func = Diff2d()
        else:
            msg = f"NOT Supported: {name}"
            raise ValueError(msg)

    # def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(output, target)
        return loss


class MCDUpdater:
    def __init__(self, cfg: Config, model: nn.Module, dataloader: TwoDomainDataLoader,
                 optimizer: BaseGenClsOptimizer, device: Device,
                 classifier_criterion: Criterion, discrepancy_criterion: Criterion):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.iter = 0
        self.device = device
        self.classifier_criterion = classifier_criterion
        self.discrepancy_criterion = discrepancy_criterion

    def _update_classifier(self, src_imgs, src_lbls):
        ### update generator and classifier by source samples
        self.optimizer.zero_grad()
        out_f1, out_f2 = self.model(src_imgs)
        loss_f1 = self.classifier_criterion(out_f1, src_lbls)
        loss_f2 = self.classifier_criterion(out_f2, src_lbls)
        loss = loss_f1 + loss_f2
        loss.backward()
        self.optimizer.generator.step()
        self.optimizer.classifier.step()
        return loss

    def _maximize_discrepancy(self, src_imgs, src_lbls, tgt_imgs):
        self.optimizer.zero_grad()
        out_src_f1, out_src_f2 = self.model(src_imgs)
        loss_src_f1 = self.classifier_criterion(out_src_f1, src_lbls)
        loss_src_f2 = self.classifier_criterion(out_src_f2, src_lbls)

        out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
        loss_discrepancy = self.discrepancy_criterion(out_tgt_f1, out_tgt_f2)
        loss = loss_src_f1 + loss_src_f2 - loss_discrepancy
        loss.backward()
        self.optimizer.classifier.step()

    def _minimize_discrepancy(self, tgt_imgs):
        self.optimizer.zero_grad()
        out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
        loss = self.discrepancy_criterion(out_tgt_f1, out_tgt_f2)
        loss.backward()
        self.optimizer.generator.step()
        return loss

    def train(self, N_repeat_generator_update):
        self.model.train()

        pbar = tqdm(self.dataloader.train)
        for batch_idx, (source, target) in enumerate(pbar):
            src_imgs = source[0].to(self.device.get())
            src_lbls = source[1].to(self.device.get())
            tgt_imgs = target[0].to(self.device.get())

            loss_c = self._update_classifier(src_imgs, src_lbls).item()
            self._maximize_discrepancy(src_imgs, src_lbls, tgt_imgs)

            loss_d = 0
            for _ in range(N_repeat_generator_update):
                loss_d += self._minimize_discrepancy(tgt_imgs).item()
            loss_d /= N_repeat_generator_update

            msg = str(
                f"Loss C: {loss_c:.6f} | "
                f"Loss D: {loss_d:.6f} | "
            )
            pbar.set_description(msg)

    def evaluate(self):
        self.model.eval()

        loss_src = 0
        loss_tgt = 0
        correct_src = 0
        correct_tgt = 0
        with torch.no_grad():
            for source, target in tqdm(self.dataloader.val):
                src_imgs = source[0].to(self.device.get())
                src_lbls = source[1].to(self.device.get())
                tgt_imgs = target[0].to(self.device.get())
                tgt_lbls = target[1].to(self.device.get())

                out_src_f1, out_src_f2 = self.model(src_imgs)
                out_src = out_src_f1 + out_src_f2
                # out_src = out_src_f1
                out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
                out_tgt = out_tgt_f1 + out_tgt_f2
                # out_tgt = out_tgt_f1

                pred_src = out_src.argmax(dim=1, keepdim=True)
                pred_tgt = out_tgt.argmax(dim=1, keepdim=True)
                correct_src += pred_src.eq(src_lbls.view_as(pred_src)).sum().item()
                correct_tgt += pred_tgt.eq(tgt_lbls.view_as(pred_tgt)).sum().item()

                loss_src += self.classifier_criterion(out_src, src_lbls).item()
                loss_tgt += self.classifier_criterion(out_tgt, tgt_lbls).item()

        loss_src /= self.dataloader.val_len
        loss_tgt /= self.dataloader.val_len

        mean_acc_src = correct_src / self.dataloader.val_len
        mean_acc_tgt = correct_tgt / self.dataloader.val_len

        msg = str(
            "Test set: \n"
            f"\tAverage loss: \n"
            f"\t\tsource: {loss_src:.4f}\n"
            f"\t\ttarget: {loss_tgt:.4f}\n"
            f"\tAccuracy: \n"
            f"\t\tsource: {correct_src:.4f} / {self.dataloader.val_len} ({100 * mean_acc_src:.2f}%)\n"
            f"\t\ttarget: {correct_tgt:.4f} / {self.dataloader.val_len} ({100 * mean_acc_tgt:.2f}%)\n"
        )
        print(msg)
        return mean_acc_src, mean_acc_tgt


def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
    )
    model = Net().to(device.get())
    optimizer = Optimizer(
        model=model,
        cfg=cfg
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        classifier_criterion=Criterion(name=Criterion.CEL),
        discrepancy_criterion=Criterion(name=Criterion.diff)
    )

    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        updater.train(N_repeat_generator_update=4)
        updater.evaluate()

    if (cfg.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


def objective(trial):
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
    )
    model = Net().to(device.get())

    cfg.optim.lr_c = trial.suggest_loguniform("lr_c", 1e-10, 1e-3)
    cfg.optim.lr_g = trial.suggest_loguniform("lr_g", 1e-10, 1e-3)
    optimizer = Optimizer(
        model=model,
        cfg=cfg.optim
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        classifier_criterion=Criterion(name=Criterion.CEL),
        discrepancy_criterion=Criterion(name=Criterion.diff)
    )

    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        updater.train(N_repeat_generator_update=4)
        mean_acc_src, mean_acc_tgt = updater.evaluate()
    error_rate = 1 - mean_acc_tgt
    return error_rate


if __name__ == '__main__':
    # train()
    import optuna

    study = optuna.create_study()
    study.optimize(objective, n_trials=2)

    ##########
    # oputna log
    ##########
    import json

    with open("./best_param.json", "w") as fw:
        json.dump(study.best_params, fw)
    print("")
    print("++++ optuna trial ++++")
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))
