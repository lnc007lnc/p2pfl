from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.node import Node
import lightning as L
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric
import torch.nn as nn
import torch
from torchvision import models
import time
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

class ResNet18CIFAR10(L.LightningModule):
    """ResNet-18 model adapted for CIFAR10 classification."""

    def __init__(
        self,
        num_classes: int = 10,
        metric: type[Metric] = Accuracy,
        lr_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        # ensure reproducibility
        set_seed(Settings.general.SEED, "pytorch")

        self.lr_rate = lr_rate
        # choose metric: binary vs multiclass
        if num_classes == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=num_classes)

        # load standard ResNet18
        model = models.resnet18(weights=None, num_classes=num_classes)
        # adapt to CIFAR10: 32×32 inputs
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if input is H×W×C, permute to C×H×W
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        logits = self.model(x)
        return F.log_softmax(logits, dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["image"].float()
        y = batch["label"]
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["image"].float()
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["image"].float()
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss


def resnet(*args, **kwargs) -> LightningModel:
    """
    Export the ResNet-18 CIFAR10 model for P2PFL.
    Accepts the same args/kwargs as ResNet18CIFAR10.
    """
    compression = kwargs.pop("compression", None)
    return LightningModel(ResNet18CIFAR10(*args, **kwargs), compression=compression)


# Start the node
node1 = Node(
    model = resnet(),
    data = P2PFLDataset.from_huggingface("p2pfl/CIFAR10"), # Get dataset from Hugging Face
    addr= f"127.0.0.1:{6668}", # Introduce a port or remove to use a random one
)
node1.start()

node2 = Node(
    model = resnet(),
    data = P2PFLDataset.from_huggingface("p2pfl/CIFAR10"),
    addr = "127.0.0.1", # Random port
)
node2.start()

node3 = Node(
    model = resnet(),
    data = P2PFLDataset.from_huggingface("p2pfl/CIFAR10"), # Get dataset from Hugging Face
    addr= f"127.0.0.1:{6669}", # Introduce a port or remove to use a random one
)
node3.start()

node2.connect(f"127.0.0.1:{6668}")
node2.connect(f"127.0.0.1:{6669}")
node3.connect(f"127.0.0.1:{6668}")
time.sleep(4)

node1.set_start_learning(rounds=10, epochs=1)

while True:
    time.sleep(1)
    #print(node2.state)
    if node2.state.round is None:
        break

# Stop the node
node1.stop()
node2.stop()
node3.stop()