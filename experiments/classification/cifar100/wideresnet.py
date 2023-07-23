# fmt: off
from pathlib import Path

import torch.nn as nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import WideResNet
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.optimization_procedures import get_procedure

# fmt: on
if __name__ == "__main__":
    args = init_args(WideResNet, CIFAR100DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"{args.version}-wideresnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR100DataModule(**vars(args))

    # model
    model = WideResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(
            f"resnet{args.arch}", "cifar100", args.version
        ),
        style="cifar",
        **vars(args),
    )

    cli_main(model, dm, root, net_name, args)