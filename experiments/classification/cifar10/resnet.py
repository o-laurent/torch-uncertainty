# fmt: off
from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure

# fmt: on
if __name__ == "__main__":
    args = init_args(ResNet, CIFAR10DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    if args.opt_temp_scaling:
        args.calibration_set = dm.get_test_set
    elif args.val_temp_scaling:
        args.calibration_set = dm.get_val_set
    else:
        args.calibration_set = None

    # model
    model = ResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(
            f"resnet{args.arch}", "cifar10", args.version
        ),
        style="cifar",
        **vars(args),
    )

    cli_main(model, dm, args.exp_dir, net_name, args)
