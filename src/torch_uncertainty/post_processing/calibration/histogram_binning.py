import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_uncertainty.post_processing import PostProcessing


class HistogramBinningScaler(PostProcessing):
    def __init__(
        self,
        model: nn.Module | None = None,
        num_bins: int = 15,
        ovr_binning: bool = True,
        eps: float = 1e-6,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Histogram Binning post-processing for calibrated probabilities.

        Histogram binning is a non-parametric calibration method that partitions
        the uncalibrated probability space into equal-width bins. For each bin,
        it computes the empirical probability of the positive class.

        Args:
            model (nn.Module): Model to calibrate. Defaults to ``None``.
            num_bins (int): Number of equal-width bins to use. Defaults to ``15``.
            ovr_binning (bool): Whether to use One-vs-Rest binning for multiclass.
                If ``True``, fits separate bins per class. If ``False``, fits
                a single set of bins shared across all classes. Defaults to ``True``.
            eps (float): Small value for stability when converting probs back to logits.
                Defaults to ``1e-6``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for
                tensor operations. Defaults to ``None``.

        References:
            [1] Obtaining calibrated probability estimates from decision trees
            and naive bayesian classifiers. In ICML 2001.
            <https://cseweb.ucsd.edu/~elkan/calibrated.pdf>`_.
        """
        super().__init__(model)

        if num_bins <= 0:
            raise ValueError(f"Number of bins must be strictly positive. Got {num_bins}.")

        self.num_bins = num_bins
        self.ovr_binning = ovr_binning
        self.device = device
        self.eps = eps

        self.num_classes: int | None = None
        self.bin_edges: Tensor | None = None
        self.bin_values: Tensor | None = None

    def fit(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> None:
        """Fit the histogram binning model to the calibration data.

        Args:
            dataloader (DataLoader): Dataloader providing the calibration data.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to ``True``.
        """
        if self.model is None or isinstance(self.model, nn.Identity):
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()

        all_logits, all_labels = self._extract_data(dataloader, progress)

        # Determine dimensionality
        if all_logits.dim() == 1 or (all_logits.dim() == 2 and all_logits.shape[1] == 1):
            probs = torch.sigmoid(all_logits).flatten()
            labels = all_labels.float().flatten()
            self.num_classes = 1
        else:
            probs = torch.softmax(all_logits, dim=-1)
            labels = all_labels
            self.num_classes = probs.shape[1]

        # Define equal-width bin edges and centers
        self.bin_edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device=self.device)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        # Extract internal boundaries for `torch.bucketize`
        boundaries = self.bin_edges[1:-1]

        if self.num_classes == 1:
            self.bin_values = torch.zeros(self.num_bins, device=self.device)
            indices = torch.bucketize(probs, boundaries)
            for b in range(self.num_bins):
                mask = indices == b
                if mask.any():
                    self.bin_values[b] = labels[mask].mean()
                else:
                    self.bin_values[b] = bin_centers[b]

        elif not self.ovr_binning:
            # Shared Multiclass: Pool all probabilities and one-hot targets
            probs_flat = probs.flatten()
            labels_one_hot = F.one_hot(labels.long(), self.num_classes).float().flatten()

            self.bin_values = torch.zeros(self.num_bins, device=self.device)
            indices = torch.bucketize(probs_flat, boundaries)
            for b in range(self.num_bins):
                mask = indices == b
                if mask.any():
                    self.bin_values[b] = labels_one_hot[mask].mean()
                else:
                    self.bin_values[b] = bin_centers[b]

        else:
            # One-vs-Rest Multiclass
            self.bin_values = torch.zeros((self.num_classes, self.num_bins), device=self.device)
            labels_one_hot = F.one_hot(labels.long(), self.num_classes).float()

            for c in range(self.num_classes):
                c_probs = probs[:, c]
                c_labels = labels_one_hot[:, c]
                indices = torch.bucketize(c_probs, boundaries)
                for b in range(self.num_bins):
                    mask = indices == b
                    if mask.any():
                        self.bin_values[c, b] = c_labels[mask].mean()
                    else:
                        self.bin_values[c, b] = bin_centers[b]

        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply Histogram Binning and return calibrated logits."""
        if self.model is None or not self.trained:
            logging.warning("Scaler not trained. Returning raw inputs.")
            return self.model(inputs)

        logits = self.model(inputs)
        boundaries = self.bin_edges[1:-1]

        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            indices = torch.bucketize(probs, boundaries)
            calib_probs = self.bin_values[indices]

        elif not self.ovr_binning:
            probs = torch.softmax(logits, dim=-1)
            indices = torch.bucketize(probs, boundaries)
            calib_probs = self.bin_values[indices]
            calib_probs /= calib_probs.sum(dim=-1, keepdim=True) + 1e-12

        else:  # OvR Strategy
            probs = torch.softmax(logits, dim=-1)
            indices = torch.bucketize(probs, boundaries)  # Shape: (N, K)

            # Vectorized mapping: gather bin_values for each class
            # note: bin_values is (K, B), indices is (N, K)
            expanded_bins = self.bin_values.unsqueeze(0).expand(indices.size(0), -1, -1)
            calib_probs = torch.gather(expanded_bins, 2, indices.unsqueeze(-1)).squeeze(-1)
            calib_probs /= calib_probs.sum(dim=-1, keepdim=True) + 1e-12

        # Convert back to pseudo-logit space
        calib_probs = calib_probs.clamp(self.eps, 1 - self.eps)
        if self.num_classes == 1:
            return torch.logit(calib_probs, eps=self.eps)
        return torch.log(calib_probs)

    def _extract_data(self, dataloader: DataLoader, progress: bool) -> tuple[Tensor, Tensor]:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, disable=not progress):
                logits = self.model(inputs.to(self.device))
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)

        if all_labels.ndim == 1:
            all_labels = all_labels.long()

        return all_logits, all_labels
