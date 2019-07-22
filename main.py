import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


from abc import ABCMeta


class Config:
    class BaseChildConfig(metaclass=ABCMeta):
        pass

    class Optim(BaseChildConfig):
        name = "sgd"
        lr = 0.01
        momentum = 0.5

    def __init__(self):
        self.seed = 1
        self.batch_size = 64
        self.test_batch_size = 1
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


class DataLoader:
    def __init__(self, device: Device, batch_size: int, test_batch_size: int, name="mnist"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device.is_cuda else {}
        DATA_ROOT = "/raid/pytorch"
        self.train = torch.utils.data.DataLoader(
            datasets.MNIST(str(DATA_ROOT), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.val = torch.utils.data.DataLoader(
            datasets.MNIST(str(DATA_ROOT), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    @property
    def train_len(self):
        return len(self.train.dataset)

    @property
    def val_len(self):
        return len(self.val.dataset)


from enum import Enum


class Optimizer:
    class SUPPORTED(Enum):
        sgd = "sgd"

    def __init__(self, model: nn.Module, cfg: Config):
        self.cfg = cfg

        if self.cfg.optim.name == self.SUPPORTED.sgd.value:
            self.module = optim.SGD(
                model.parameters(),
                lr=cfg.optim.lr, momentum=cfg.optim.momentum
            )
        else:
            msg = f"NOT Supported: {self.cfg.optim.name}"
            raise ValueError(msg)

    def zero_grad(self):
        self.module.zero_grad()

    def step(self):
        self.module.step()


class Criterion:
    def __call__(self, output: torch.Tensor, target: torch.Tensor,
                 reduction: str = "mean") -> torch.Tensor:
        loss = F.nll_loss(output, target, reduction=reduction)
        return loss


class ClassfierUpdater:
    def __init__(self, cfg: Config, model: nn.Module, dataloader: DataLoader,
                 optimizer: Optimizer, device: Device, criterion: Criterion):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.iter = 0
        self.device = device
        self.criterion = criterion

    def train(self):
        self.model.train()

        pbar = tqdm(self.dataloader.train)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device.get())
            target = target.to(self.device.get())

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            self.optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.6f}")

    def evaluate(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(self.dataloader.val):
                data = data.to(self.device.get())
                target = target.to(self.device.get())

                output = self.model(data)
                loss = self.criterion(output, target, reduction="sum")
                test_loss += loss.item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataloader.val.dataset)
        mean_acc = correct / self.dataloader.val_len

        msg = str(
            "Test set: \n"
            f"\tAverage loss: {test_loss:.4f}\n"
            f"\tAccuracy    : {correct:.4f} / {self.dataloader.val_len} ({100 * mean_acc:.2f}%)"
        )
        print(msg)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = DataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        name="mnist"
    )
    model = Net().to(device.get())
    optimizer = Optimizer(
        model=model,
        cfg=cfg
    )
    criterion = Criterion()

    updater = ClassfierUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device, criterion=criterion
    )

    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        updater.train()
        updater.evaluate()

    if (cfg.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
