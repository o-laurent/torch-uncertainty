from pathlib import Path

import torch
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torchvision import tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets import NYUv2
from torch_uncertainty.transforms import RandomRescale
from torch_uncertainty.utils.misc import create_train_val_split


class NYUv2DataModule(AbstractDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        max_depth: float = 10.0,
        crop_size: _size_2_t = (416, 544),
        inference_size: _size_2_t = (416, 544),
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""Depth DataModule for the NYUv2 dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch.
            max_depth (float, optional): Maximum depth value. Defaults to
                ``80.0``.
            crop_size (sequence or int, optional): Desired input image and
                depth mask sizes during training. If :attr:`crop_size` is an
                int instead of sequence like :math:`(H, W)`, a square crop
                :math:`(\text{size},\text{size})` is made. If provided a sequence
                of length :math:`1`, it will be interpreted as
                :math:`(\text{size[0]},\text{size[1]})`. Defaults to ``(416, 544)``.
            inference_size (sequence or int, optional): Desired input image and
                depth mask sizes during inference. If size is an int,
                smaller edge of the images will be matched to this number, i.e.,
                :math:`\text{height}>\text{width}`, then image will be rescaled to
                :math:`(\text{size}\times\text{height}/\text{width},\text{size})`.
                Defaults to ``(416, 544)``.
            val_split (float or None, optional): Share of training samples to use
                for validation. Defaults to ``None``.
            num_workers (int, optional): Number of dataloaders to use. Defaults to
                ``1``.
            pin_memory (bool, optional):  Whether to pin memory. Defaults to
                ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset = NYUv2
        self.max_depth = max_depth
        self.crop_size = _pair(crop_size)
        self.inference_size = _pair(inference_size)

        self.train_transform = v2.Compose(
            [
                RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                v2.RandomCrop(
                    size=self.crop_size,
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 0, tv_tensors.Mask: float("nan")},
                ),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.Resize(size=self.inference_size, antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(
            root=self.root,
            split="train",
            max_depth=self.max_depth,
            download=True,
        )
        self.dataset(
            root=self.root, split="val", max_depth=self.max_depth, download=True
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                root=self.root,
                max_depth=self.max_depth,
                split="train",
                transforms=self.train_transform,
            )

            if self.val_split is not None:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    root=self.root,
                    max_depth=self.max_depth,
                    split="val",
                    transforms=self.test_transform,
                )

        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                max_depth=self.max_depth,
                split="val",
                transforms=self.test_transform,
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
