from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts import (
    SupervisedTaskSpec,
    SupervisedFeatureBundle,
)
from ..common.losses import (
    compute_balanced_class_loss_config,
    compute_balanced_rank_label_loss_config,
)
from ..common.prediction import TorchSequencePredictionMixin, selection_auroc_from_logits
from ...tasks import (
    default_binary_task_spec as _default_binary_task_spec,
)
from ..common.torch_runtime import (
    set_torch_random_seeds,
    set_torch_threads,
)
from .data import (
    CNNChannelLayout,
    CNNFeatureTensors,
    CNNLayerVectorConfig,
    CNNNormalizationStats,
    build_per_layer_vectors,
    pad_layer_sequence_batch,
)
from .network import (
    _CNNLayerSequenceClassifier,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - soft dependency
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None


CNN_MAX_EPOCHS = 40
CNN_BATCH_SIZE = 16
CNN_PATIENCE = 5


def _require_torch() -> None:
    if torch is None or F is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ModuleNotFoundError("cnn_1d requires the optional 'torch' dependency, but torch is not installed")


class CNN1DSupervisedModel(TorchSequencePredictionMixin):
    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
        normalize_input_features: bool = True,
        device: str | None = None,
    ) -> None:
        _require_torch()
        if int(conv_channels) <= 0:
            raise ValueError("cnn_1d conv_channels must be positive")
        if int(num_conv_layers) <= 0:
            raise ValueError("cnn_1d num_conv_layers must be positive")
        if int(kernel_size) <= 0:
            raise ValueError("cnn_1d kernel_size must be positive")
        if int(stride) <= 0:
            raise ValueError("cnn_1d stride must be positive")
        if int(dilation) <= 0:
            raise ValueError("cnn_1d dilation must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("cnn_1d dropout must be in [0, 1)")
        if bool(class_weight_loss) and bool(rank_label_weight_loss):
            raise ValueError("cnn_1d supports either class_weight_loss or rank_label_weight_loss, not both")
        self.layer_vector_config = CNNLayerVectorConfig(
            conv_channels=int(conv_channels),
            num_conv_layers=int(num_conv_layers),
            kernel_size=int(kernel_size),
            stride=int(stride),
            dilation=int(dilation),
            dropout=float(dropout),
            use_residual=bool(use_residual),
            normalization=str(normalization),
            pooling=str(pooling),
            include_total_layer_count=bool(include_total_layer_count),
            depth_feature_mode=str(depth_feature_mode),
        )
        self.conv_channels = int(conv_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.normalization = str(normalization)
        self.pooling = str(pooling)
        self.include_total_layer_count = bool(include_total_layer_count)
        self.depth_feature_mode = str(depth_feature_mode)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight_loss = bool(class_weight_loss)
        self.rank_label_weight_loss = bool(rank_label_weight_loss)
        self.normalize_input_features = bool(normalize_input_features)
        self.device = str(device) if device is not None else "cuda"
        self.device_ = torch.device(self.device)
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cnn_1d was configured for CUDA, but CUDA is not available")
        self.model_: _CNNLayerSequenceClassifier | None = None
        self.normalization_stats_: CNNNormalizationStats | None = None
        self.channel_layout_: CNNChannelLayout | None = None
        self.channel_mean_: np.ndarray | None = None
        self.channel_std_: np.ndarray | None = None
        self.input_channels_: int | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "cnn_1d"
        self._fit_summary: dict[str, Any] = {}

    def _set_random_seeds(self) -> None:
        set_torch_random_seeds(self.random_state)

    def _set_threads(self, n_jobs: int | None) -> None:
        set_torch_threads(n_jobs)

    def _prepare_numpy_inputs(self, bundle: SupervisedFeatureBundle) -> CNNFeatureTensors:
        vector_batch = build_per_layer_vectors(
            bundle,
            normalization_stats=self.normalization_stats_,
            include_total_layer_count=bool(self.include_total_layer_count),
            depth_feature_mode=str(self.depth_feature_mode),
            normalize_continuous_features=bool(self.normalize_input_features),
        )
        if self.channel_layout_ is None:
            self.channel_layout_ = vector_batch.channel_layout
        elif self.channel_layout_ != vector_batch.channel_layout:
            raise ValueError("cnn_1d encountered an incompatible layer-sequence channel layout between bundles")

        if self.normalization_stats_ is None:
            self.normalization_stats_ = vector_batch.normalization_stats
            self.channel_mean_ = np.asarray(
                vector_batch.normalization_stats.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                vector_batch.normalization_stats.channel_std,
                dtype=np.float32,
            )
        elif self.channel_mean_ is None or self.channel_std_ is None:
            self.channel_mean_ = np.asarray(
                self.normalization_stats_.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                self.normalization_stats_.channel_std,
                dtype=np.float32,
            )

        padded_batch = pad_layer_sequence_batch(vector_batch.sequences)
        self.input_channels_ = int(padded_batch.inputs.shape[2])
        return padded_batch

    def _build_model(self, *, input_channels: int) -> _CNNLayerSequenceClassifier:
        _require_torch()
        model = _CNNLayerSequenceClassifier(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
        )
        self.input_channels_ = int(input_channels)
        return model.to(self.device_)

    def _logits_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                batch_inputs = batch_inputs.to(self.device_, non_blocking=True)
                batch_layer_mask = batch_layer_mask.to(self.device_, non_blocking=True)
                logits = self.model_(batch_inputs, batch_layer_mask)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        if not outputs:
            if self.task_spec.is_binary:
                return np.asarray([], dtype=np.float64)
            return np.asarray([], dtype=np.float64).reshape(0, self.task_spec.n_classes)
        combined = np.concatenate(outputs, axis=0)
        if self.task_spec.is_binary:
            return combined.reshape(-1)
        return np.asarray(combined, dtype=np.float64)

    def _features_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                batch_inputs = batch_inputs.to(self.device_, non_blocking=True)
                batch_layer_mask = batch_layer_mask.to(self.device_, non_blocking=True)
                embeddings = self.model_.extract_features(batch_inputs, batch_layer_mask)
                outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
        if not outputs:
            embedding_dim = 0
            if self.model_ is not None:
                embedding_dim = int(self.model_.aggregator.embedding_dim)
            return np.asarray([], dtype=np.float32).reshape(0, embedding_dim)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        rank_labels: np.ndarray | None = None,
    ) -> "CNN1DSupervisedModel":
        _require_torch()
        assert torch is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        train_tensors = self._prepare_numpy_inputs(bundle)
        if self.task_spec.is_binary:
            y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
        else:
            y_train = np.asarray(labels, dtype=np.int64).reshape(-1)
        if train_tensors.inputs.shape[0] != y_train.shape[0]:
            raise ValueError("cnn_1d training features/labels length mismatch")
        rank_label_loss_config: dict[str, Any] | None = None
        rank_label_sample_weights = np.ones(y_train.shape[0], dtype=np.float32)
        if self.rank_label_weight_loss:
            if rank_labels is None:
                raise ValueError("cnn_1d rank_label_weight_loss requires rank_labels")
            rank_labels_np = np.asarray(rank_labels, dtype=np.int64).reshape(-1)
            if rank_labels_np.shape[0] != y_train.shape[0]:
                raise ValueError("cnn_1d rank_labels must have the same length as labels")
            rank_label_sample_weights, rank_label_loss_config = compute_balanced_rank_label_loss_config(
                y_train,
                rank_labels_np,
                task_spec=self.task_spec,
            )

        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
            torch.from_numpy(y_train),
            torch.from_numpy(rank_label_sample_weights),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
            pin_memory=(self.device_.type == "cuda"),
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_dataset = TensorDataset(
                torch.from_numpy(valid_tensors.inputs),
                torch.from_numpy(valid_tensors.layer_mask),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                shuffle=False,
                pin_memory=(self.device_.type == "cuda"),
            )

        self.model_ = self._build_model(input_channels=int(train_tensors.inputs.shape[2]))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        class_loss_config: dict[str, Any] | None = None
        if self.class_weight_loss:
            class_loss_config = compute_balanced_class_loss_config(
                y_train,
                task_spec=self.task_spec,
            )

        if self.task_spec.is_binary:
            pos_weight = None
            if class_loss_config is not None:
                pos_weight = torch.tensor(
                    float(class_loss_config["binary_pos_weight"]),
                    dtype=torch.float32,
                    device=self.device_,
                )
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        else:
            class_weight_tensor = None
            if class_loss_config is not None:
                class_weight_tensor = torch.tensor(
                    np.asarray(class_loss_config["class_weights"], dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device_,
                )
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )

        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch_idx in range(self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            train_count = 0.0
            for batch_inputs, batch_layer_mask, batch_labels, batch_sample_weights in train_loader:
                batch_inputs = batch_inputs.to(self.device_, non_blocking=True)
                batch_layer_mask = batch_layer_mask.to(self.device_, non_blocking=True)
                batch_labels = batch_labels.to(self.device_, non_blocking=True)
                batch_sample_weights = batch_sample_weights.to(self.device_, non_blocking=True)
                optimizer.zero_grad()
                logits = self.model_(batch_inputs, batch_layer_mask)
                loss_raw = loss_fn(logits, batch_labels)
                if self.rank_label_weight_loss:
                    weights = batch_sample_weights.to(dtype=loss_raw.dtype)
                    loss_numerator = torch.sum(loss_raw.reshape(-1) * weights.reshape(-1))
                    loss_denominator = torch.clamp(torch.sum(weights), min=1.0)
                    loss = loss_numerator / loss_denominator
                else:
                    loss = loss_raw
                loss.backward()
                optimizer.step()
                batch_size = int(batch_labels.shape[0])
                if self.rank_label_weight_loss:
                    train_loss_sum += float(loss_numerator.item())
                    train_count += float(loss_denominator.item())
                else:
                    train_loss_sum += float(loss.item()) * batch_size
                    train_count += float(batch_size)

            train_loss = train_loss_sum / max(1, train_count)
            metric = -train_loss
            if valid_loader is not None and y_valid is not None:
                valid_logits = self._logits_from_loader(valid_loader)
                metric = selection_auroc_from_logits(
                    y_valid,
                    valid_logits,
                    task_spec=self.task_spec,
                )

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "train_loss": float(train_loss),
                    "selection_metric": float(metric),
                    "selection_metric_name": str(self.task_spec.selection_metric_name),
                }
            )

            if metric > best_metric:
                best_metric = float(metric)
                best_epoch = int(epoch_idx)
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("cnn_1d training did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.to(self.device_)
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
            "class_weight_loss": bool(self.class_weight_loss),
            "class_loss_weights": class_loss_config,
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "rank_label_loss_weights": rank_label_loss_config,
            "device": str(self.device_),
        }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
            pin_memory=(self.device_.type == "cuda"),
        )
        return self._logits_from_loader(loader)

    def extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
            pin_memory=(self.device_.type == "cuda"),
        )
        return self._features_from_loader(loader)

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {}

    def save(self, path: Path) -> None:
        _require_torch()
        assert torch is not None
        if (
            self.model_ is None
            or self.channel_mean_ is None
            or self.channel_std_ is None
            or self.channel_layout_ is None
        ):
            raise RuntimeError("cnn_1d model has not been fit")
        payload = {
            "backend": self.backend_name_,
            "config": {
                "conv_channels": int(self.conv_channels),
                "num_conv_layers": int(self.num_conv_layers),
                "kernel_size": int(self.kernel_size),
                "stride": int(self.stride),
                "dilation": int(self.dilation),
                "dropout": float(self.dropout),
                "use_residual": bool(self.use_residual),
                "normalization": str(self.normalization),
                "pooling": str(self.pooling),
                "include_total_layer_count": bool(self.include_total_layer_count),
                "depth_feature_mode": str(self.depth_feature_mode),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
                "random_state": int(self.random_state),
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "patience": int(self.patience),
                "class_weight_loss": bool(self.class_weight_loss),
                "rank_label_weight_loss": bool(self.rank_label_weight_loss),
                "normalize_input_features": bool(self.normalize_input_features),
                "input_channels": int(self.input_channels_ or 0),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "channel_layout": asdict(self.channel_layout_),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.model_.state_dict().items()
            },
            "normalization": {
                "channel_mean": np.asarray(self.channel_mean_, dtype=np.float32),
                "channel_std": np.asarray(self.channel_std_, dtype=np.float32),
            },
            "classes": np.asarray(self.classes_, dtype=np.int32),
            "class_names": list(self.class_names_),
            "task": self.task_spec.to_dict(),
            "fit_summary": dict(self._fit_summary),
        }
        payload.update(self._checkpoint_extra_payload())
        torch.save(payload, Path(path).expanduser().resolve())