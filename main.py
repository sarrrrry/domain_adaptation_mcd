import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from mcd.networks.base_net_mcd import BaseNetMCD


class Net_FeatureExtractore(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
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
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net(BaseNetMCD):
    def __init__(self, ckpt: str = None):
        super().__init__()
        self.generator = Net_FeatureExtractore()
        self.classifier_f1 = Net_Classifier()
        self.classifier_f2 = Net_Classifier()

    def forward(self, x):
        x = self.generator(x)
        x = x.view(x.shape[0], -1)
        x_f1 = self.classifier_f1(x)
        x_f2 = self.classifier_f2(x)
        return x_f1, x_f2


from abc import ABCMeta


class Config:
    class BaseChildConfig(metaclass=ABCMeta):
        pass

    class Optim(BaseChildConfig):
        # c_name = "sgd"
        # c_lr = 0.000000002
        c_name = "adam"
        c_lr = 0.0002
        c_weight_decay = 0.0005
        g_name = "adam"
        g_lr = 0.0002
        g_weight_decay = 0.0005

    def __init__(self):
        self.N_repeat_genrator_update = 5  # from optuna
        self.multiply_loss_discrepancy = 1.8  # from optuna
        self.seed = 1
        # self.batch_size = 64
        # self.batch_size = 10000
        # self.batch_size = 4056
        self.batch_size = 128
        # self.test_batch_size = 64
        self.test_batch_size = self.batch_size
        self.epochs = 10000
        self.log_interval = 10
        self.save_model = True
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
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, i):
        return self.source[i], self.target[i]

    def __len__(self):
        return min(len(self.source), len(self.target))


class ConcatGray2RGB:
    def __call__(self, img: torch.Tensor):
        if not len(img.shape) == 3:
            raise ValueError
        img = torch.cat((img, img, img), 0)
        return img
class TwoDomainDataLoader:
    def __init__(self, device: Device, batch_size: int, test_batch_size: int, seed=1):
        img_size = (32, 32)
        kwargs = {'num_workers': 1, 'pin_memory': True} if device.is_cuda else {}

        DATA_ROOT = "/raid/pytorch"
        source = datasets.SVHN(
            str(DATA_ROOT), split="train", download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )
            ])
        )
        target = datasets.MNIST(
            str(DATA_ROOT), train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                ConcatGray2RGB(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )
            ])
        )
        train_dataset = ConcatDataset(source, target)
        val_dataset = ConcatDataset(source, target)

        def worker_init_fn(worker_id):
            random.seed(seed)
        self.train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            worker_init_fn=worker_init_fn, **kwargs)
        self.val = torch.utils.data.DataLoader(
            val_dataset, batch_size=test_batch_size,
            shuffle=True, worker_init_fn=worker_init_fn, **kwargs)

    @property
    def train_len(self):
        return len(self.train.dataset)

    @property
    def val_len(self):
        return len(self.val.dataset)


import torch.optim.lr_scheduler


class LR_Scheduler:
    class NoneScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self):
            pass

        def step(self):
            pass

    EXP = "exp"
    STP = "step"
    COS = "cos"

    def __init__(self, optimizer, name):
        lrs = torch.optim.lr_scheduler
        gen = optimizer.generator
        cl1 = optimizer.classifier_f1
        cl2 = optimizer.classifier_f2
        if name == self.EXP:
            self.generator = lrs.ExponentialLR(gen, gamma=0.9)
            self.classifier_f1 = lrs.ExponentialLR(cl1, gamma=0.9)
            self.classifier_f2 = lrs.ExponentialLR(cl2, gamma=0.9)
        elif name == self.STP:
            self.generator = lrs.StepLR(gen, step_size=30, gamma=0.9)
            self.classifier_f1 = lrs.StepLR(cl1, step_size=30, gamma=0.9)
            self.classifier_f2 = lrs.StepLR(cl2, step_size=30, gamma=0.9)
        elif name == self.COS:
            self.generator = lrs.CosineAnnealingLR(gen, 5e-6, 1e-10)
            self.classifier_f1 = lrs.CosineAnnealingLR(cl1, 5e-6, 1e-10)
            self.classifier_f2 = lrs.CosineAnnealingLR(cl2, 5e-6, 2e-20)
        else:
            self.generator = self.NoneScheduler()
            self.classifier_f1 = self.NoneScheduler()
            self.classifier_f2 = self.NoneScheduler()

    def step(self):
        self.generator.step()
        self.classifier_f1.step()
        self.classifier_f2.step()

class Optimizer:
    supported = {
        "sgd": optim.SGD,
        "adam": optim.Adam
    }
    SCHEDULER_NONE = None
    SCHEDULER_EXP = LR_Scheduler.EXP
    SCHEDULER_STP = LR_Scheduler.STP
    SCHEDULER_COS = LR_Scheduler.COS

    def __init__(self, model_params, cfg: Config.Optim, scheduler: str = None):
        model_params = model_params

        if not cfg.c_name in self.supported.keys():
            raise ValueError
        if not cfg.g_name in self.supported.keys():
            raise ValueError

        self.generator = self.supported[cfg.g_name](
            # model_params[0], momentum=0.8,
            model_params[0],
            lr=cfg.g_lr, weight_decay=cfg.g_weight_decay
        )
        self.classifier_f1 = self.supported[cfg.c_name](
            # model_params[1], momentum=0.8,
            model_params[1],
            lr=cfg.c_lr, weight_decay=cfg.c_weight_decay
        )
        self.classifier_f2 = self.supported[cfg.c_name](
            # model_params[2], momentum=0.8,
            model_params[2],
            lr=cfg.c_lr, weight_decay=cfg.c_weight_decay
        )

        self.scheduler = LR_Scheduler(self, name=scheduler)

    def zero_grad(self):
        self.generator.zero_grad()
        self.classifier_f1.zero_grad()
        self.classifier_f2.zero_grad()

    def step_all(self):
        self.step_generator()
        self.step_classifier()

    def step_classifier(self):
        self.classifier_f1.step()
        self.classifier_f2.step()

    def step_generator(self):
        self.generator.step()

    def step_epoch(self):
        self.scheduler.step()

    @property
    def lr(self):
        gen = self.generator.param_groups[0]["lr"]
        cl1 = self.classifier_f1.param_groups[0]["lr"]
        cl2 = self.classifier_f2.param_groups[0]["lr"]
        ret = {
            "generator": gen,
            "classifier_f1": cl1,
            "classifier_f2": cl2,
        }
        return ret



class Diff2d(nn.Module):
    def __init__(self, weight=None):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        # return torch.mean(torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1)))
        return torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1))


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
            self.loss_func = nn.CrossEntropyLoss(reduction=reduction)
        elif name == self.diff:
            self.loss_func = Diff2d()
        else:
            msg = f"NOT Supported: {name}"
            raise ValueError(msg)

    # def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(output, target)
        return loss


class ClassifierMetrics:
    def update(self, key: str, idx: int, value: torch.Tensor):
        pass

class MCDUpdater:
    def __init__(self, cfg: Config, model: BaseNetMCD,
                 dataloader: TwoDomainDataLoader,
                 device: Device,
                 optimizer: Optimizer,
                 criterion_c: Criterion,
                 criterion_g: Criterion,
                 metrics: ClassifierMetrics):
        """
        c ≡ classifier
        g ≡ generator
        """
        self.model = model
        self.dataloader = dataloader
        self.iter = 0
        self.device = device
        self.optimizer = optimizer
        self.criterion_c = criterion_c
        self.criterion_g = criterion_g
        self.multiply_loss_discrepancy = cfg.multiply_loss_discrepancy
        self.metrics = metrics
        self.current_idx = 0

    def train(self, N_repeat_generator_update: int) -> None:
        self.model.train()

        print(self.optimizer.lr)
        pbar = tqdm(self.dataloader.train)
        for batch_idx, (source, target) in enumerate(pbar):
            src_imgs = source[0].to(self.device.get())
            src_lbls = source[1].to(self.device.get())
            tgt_imgs = target[0].to(self.device.get())

            loss_c = self._update_classifier(src_imgs, src_lbls).item()
            self._maximize_discrepancy(src_imgs, src_lbls, tgt_imgs)
            loss_d = self._minimize_discrepancy(tgt_imgs, N_repeat=N_repeat_generator_update).item()
            loss_d /= N_repeat_generator_update

            msg = str(
                f"Loss C: {loss_c:.6f} | "
                f"Loss D: {loss_d:.6f} | "
            )
            if batch_idx % 10 == 0:
                pbar.set_description(msg)

            self.current_idx += 1
        self.optimizer.step_epoch()
        self.model.save("latest_model.ckpt")
        # self.model.save("latest_model_lr_2e-8.ckpt")

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

                loss_src += self.criterion_c(out_src, src_lbls).item()
                loss_tgt += self.criterion_c(out_tgt, tgt_lbls).item()

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
            f"\t\tsource: {int(correct_src)} / {self.dataloader.val_len} ({100 * mean_acc_src:.2f}%)\n"
            f"\t\ttarget: {int(correct_tgt)} / {self.dataloader.val_len} ({100 * mean_acc_tgt:.2f}%)\n"
        )
        print(msg)
        return mean_acc_src, mean_acc_tgt

    def _update_classifier(self, src_imgs, src_lbls):
        ### update generator and classifier by source samples
        self.optimizer.zero_grad()
        out_f1, out_f2 = self.model(src_imgs)
        loss_f1 = self.criterion_c(out_f1, src_lbls)
        loss_f2 = self.criterion_c(out_f2, src_lbls)
        loss = loss_f1 + loss_f2
        loss.backward()
        self.optimizer.step_all()
        # self.metrics.update(key="loss_c", idx=self.current_idx, value=loss.item())
        return loss

    def _maximize_discrepancy(self, src_imgs, src_lbls, tgt_imgs):
        self.optimizer.zero_grad()
        out_src_f1, out_src_f2 = self.model(src_imgs)
        loss_src_f1 = self.criterion_c(out_src_f1, src_lbls).mean()
        loss_src_f2 = self.criterion_c(out_src_f2, src_lbls).mean()

        out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
        loss_discrepancy = self.criterion_g(out_tgt_f1, out_tgt_f2).mean()
        loss_discrepancy *= self.multiply_loss_discrepancy
        loss = loss_src_f1 + loss_src_f2 - loss_discrepancy
        loss.backward()
        self.optimizer.step_classifier()
        # self.metrics.update(key="loss_max_d", idx=self.current_idx, value=loss.item())

    def _minimize_discrepancy(self, tgt_imgs, N_repeat):
        self.optimizer.zero_grad()
        loss = torch.zeros((1), requires_grad=True).to(self.device.get())
        for _ in range(N_repeat):
            out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
            loss_discrepancy = self.criterion_g(out_tgt_f1, out_tgt_f2).mean()
            loss_discrepancy *= self.multiply_loss_discrepancy
            loss += loss_discrepancy
        loss.backward()
        self.optimizer.step_generator()
        # self.metrics.update(key="loss_min_d", idx=self.current_idx, value=loss.item())
        return loss

def train(seed):
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        seed=seed
    )
    model = Net()
    # model.load("mnist_cnn.pt")
    model.load("latest_model.ckpt")
    model.to(device.get())
    optimizer = Optimizer(
        model_params=model.parameters(),
        cfg=cfg.optim,
        # scheduler=Optimizer.SCHEDULER_COS
        scheduler=Optimizer.SCHEDULER_NONE
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        criterion_c=Criterion(name=Criterion.CEL, reduction="sum"),
        criterion_g=Criterion(name=Criterion.diff),
        metrics=ClassifierMetrics(),
    )

    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        updater.train(N_repeat_generator_update=cfg.N_repeat_genrator_update)
        updater.evaluate()

    if (cfg.save_model):
        ckpt_path = "mnist_cnn.pt"
        model.save(ckpt_path)


import optuna


def objective(trial: optuna.trial.Trial):
    seed = 1

    cfg = Config()
    cfg.seed = seed
    # cfg.epochs = 1
    cfg.epochs = 2

    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        seed=seed
    )

    model = Net()
    # model.load("mnist_cnn.pt")
    model.to(device.get())

    cfg.optim.c_weight_decay = trial.suggest_uniform("c_wd", 0., 1.)
    cfg.optim.g_weight_decay = trial.suggest_uniform("g_wd", 0., 1.)
    cfg.multiply_loss_discrepancy = trial.suggest_uniform("mld", 0, 2)
    # trial_opt = list(Optimizer.supported.keys())
    # cfg.optim.c_name = trial.suggest_categorical("c_opt", trial_opt)
    # cfg.optim.g_name = trial.suggest_categorical("g_opt", trial_opt)
    cat = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    cfg.optim.c_lr = trial.suggest_categorical("c_lr", cat)
    cfg.optim.g_lr = trial.suggest_categorical("g_lr", cat)
    # cfg.optim.c_lr = trial.suggest_loguniform("c_lr", 0.0001, 0.01)
    # cfg.optim.g_lr = trial.suggest_loguniform("g_lr", 0.00001, 0.001)
    optimizer = Optimizer(
        model_params=model.parameters(),
        cfg=cfg.optim,
        scheduler=Optimizer.SCHEDULER_STP
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        criterion_c=Criterion(name=Criterion.CEL),
        criterion_g=Criterion(name=Criterion.diff),
        metrics=ClassifierMetrics()
    )

    mean_acc = []
    N_rgu = trial.suggest_int("Nrepeat", 1, 10)
    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        updater.train(N_repeat_generator_update=N_rgu)
        _, acc = updater.evaluate()
        # mean_acc.append(acc)
    # ret = mean_acc[0] - mean_acc[1]
    ret = 1 - acc
    return ret

if __name__ == '__main__':
    import random
    import numpy as np

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    isOptuna = False
    # isOptuna = True

    if not isOptuna:
        train(seed)
    else:

        # study = optuna.create_study(direction="maximize")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, n_jobs=20)

        ##########
        # oputna log
        ##########
        import json

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
        with open("./best_param.json", "w") as fw:
            json.dump(study.best_params, fw)
