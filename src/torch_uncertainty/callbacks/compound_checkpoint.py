from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor


class CompoundCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        compound_metric_dict: dict,
        dirpath: str | Path | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] = False,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ) -> None:
        r"""Save the checkpoints maximizing or minimizing a given linear form on the metric values.

        Args:
            compound_metric_dict (dict): A dictionary mapping metric names (key) to their
                corresponding factors (value) in the linear form:

                .. math:: \sum_{i} \text{metric}_i \times \text{value}_i

            dirpath (str | Path | None, optional): The directory to save the checkpoints in.
                Defaults to ``None``.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            save_last (bool | Literal["link"], optional): Whether to save the last checkpoint.
                Defaults to ``False``.
            save_top_k (int, optional): The number of best checkpoints to save. Defaults to ``1``.
            save_weights_only (bool, optional): Whether to save only the weights. Defaults to
                ``False``.
            mode (str, optional): The mode to optimize the compound metric. Defaults to ``"min"``.
            every_n_train_steps (int | None, optional): The number of training steps to wait
                between saving checkpoints. Defaults to ``None``.
            train_time_interval (timedelta | None, optional): The time interval to wait between
                saving checkpoints. Defaults to ``None``.
            every_n_epochs (int | None, optional): The number of epochs to wait between saving
                checkpoints. Defaults to ``None``.
            save_on_train_epoch_end (bool | None, optional): Whether to save the checkpoint at the
                end of each training epoch. Defaults to ``None``.
            enable_version_counter (bool, optional): Whether to enable the version counter for the
                saved checkpoints. Defaults to ``True``.
        """
        self.compound_metric_dict = compound_metric_dict
        super().__init__(
            dirpath=dirpath,
            filename="epoch={epoch}-step={step}-compound={compound_metric:.3f}",
            monitor="compound_metric",
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def _monitor_candidates(self, trainer: Trainer) -> dict[str, Tensor]:
        monitor_candidates = super()._monitor_candidates(trainer)
        result = torch.tensor(
            0.0, dtype=torch.float32, device=next(iter(monitor_candidates.values())).device
        )
        for metric, factor in self.compound_metric_dict.items():
            result += factor * monitor_candidates[metric].to(
                dtype=result.dtype, device=result.device
            )
        monitor_candidates["compound_metric"] = result
        return monitor_candidates
