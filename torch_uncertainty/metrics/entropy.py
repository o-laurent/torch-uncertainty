from typing import Any, Literal, Optional

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Entropy(Metric):
    """
    The Shannon Entropy to estimate the confidence of the estimators.
    A higher entropy means a lower confidence.

    TODO: _docstring_
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor) -> None:
        """
        Update the current entropy with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): A probability tensor of shape
                (num_estimators, batch, num_classes) or
                (batch, num_classes)
        """
        if len(probs.shape) == 2:
            batch_size = probs.size(0)
        else:
            batch_size = probs.size(1)

        entropy = torch.special.entr(probs).sum(dim=-1)

        if len(probs.shape) == 3:
            entropy = entropy.mean(dim=0)

        if self.reduction is None or self.reduction == "none":
            self.values.append(entropy)
        else:
            self.values += entropy.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute the final entropy based on inputs passed to ``update``.

        Returns:
            torch.Tensor: The final value(s) for the Entropy
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None
            return values
