from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score

from .interfaces import (
    SupervisedTaskSpec,
    SupervisedFeatureBundle,
)
from .cnn import (
    CNNLayerVectorConfig,
    _require_torch,
    _default_binary_task_spec,
    _task_spec_from_payload,
    _Conv1DBlock,
    CNN_MAX_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_PATIENCE,
    build_per_layer_vectors,
    masked_max_pool,
    masked_mean_pool,
    pad_layer_sequence_batch,
    CNNNormalizationStats,
    CNNChannelLayout,
    CNNFeatureTensors,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - soft dependency
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None

from dataclasses import asdict


def _resolve_torch_device(device: str | None = None) -> "torch.device":
    _require_torch()
    assert torch is not None
    if device is None or str(device).strip() == "" or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("autoencoder_knn requested a CUDA device, but CUDA is not available")
    return resolved


class _ConvAutoencoderEncoder(nn.Module):
    """Encoder that mirrors CNN1D convolution blocks and pooling."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.pooling = str(config.pooling)

        blocks: list[_Conv1DBlock] = []
        in_channels = int(input_dim)
        for _ in range(int(config.num_conv_layers)):
            blocks.append(
                _Conv1DBlock(
                    input_channels=int(in_channels),
                    output_channels=int(config.conv_channels),
                    kernel_size=int(config.kernel_size),
                    stride=int(config.stride),
                    dilation=int(config.dilation),
                    dropout=float(config.dropout),
                    normalization=str(config.normalization),
                    use_residual=bool(config.use_residual),
                )
            )
            in_channels = int(config.conv_channels)
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = int(config.conv_channels)
        self.embedding_dim = (
            int(config.conv_channels) * 2
            if str(config.pooling) == "mean_max"
            else int(config.conv_channels)
        )

    def encode_sequence(
        self,
        inputs: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = inputs.transpose(1, 2)
        current_mask = layer_mask
        for block in self.blocks:
            hidden, current_mask = block(hidden, current_mask)
        hidden = hidden * current_mask.unsqueeze(1).to(dtype=hidden.dtype)
        return hidden, current_mask

    def pool_features(self, hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return masked_mean_pool(hidden, layer_mask)
        if self.pooling == "max":
            return masked_max_pool(hidden, layer_mask)
        return torch.cat(
            [
                masked_mean_pool(hidden, layer_mask),
                masked_max_pool(hidden, layer_mask),
            ],
            dim=1,
        )

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        hidden, current_mask = self.encode_sequence(inputs, layer_mask)
        return self.pool_features(hidden, current_mask)


class _ConvAutoencoderDecoder(nn.Module):
    """Convolutional decoder that reconstructs the sequence from pooled embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_channels: int,
        output_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        self.embedding_to_channels: nn.Module
        if int(embedding_dim) == int(latent_channels):
            self.embedding_to_channels = nn.Identity()
        else:
            self.embedding_to_channels = nn.Linear(int(embedding_dim), int(latent_channels))

        decoder_layers: list[nn.Module] = []
        for _ in range(int(config.num_conv_layers)):
            decoder_layers.append(
                nn.ConvTranspose1d(
                    int(latent_channels),
                    int(latent_channels),
                    kernel_size=int(config.kernel_size),
                    stride=1,
                    padding=max(0, (int(config.kernel_size) - 1) * int(config.dilation) // 2),
                    dilation=int(config.dilation),
                )
            )
            if str(config.normalization) == "layernorm":
                decoder_layers.append(nn.LayerNorm(int(latent_channels)))
            elif str(config.normalization) == "batchnorm":
                decoder_layers.append(nn.BatchNorm1d(int(latent_channels)))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(float(config.dropout)))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.output_projection = nn.Conv1d(int(latent_channels), int(output_dim), kernel_size=1)

    def forward(self, embedding: torch.Tensor, target_length: int) -> torch.Tensor:
        channels = self.embedding_to_channels(embedding)
        hidden = channels.unsqueeze(-1).expand(-1, -1, int(target_length))
        for layer in self.decoder_layers:
            if isinstance(layer, nn.LayerNorm):
                hidden = layer(hidden.transpose(1, 2)).transpose(1, 2)
            else:
                hidden = layer(hidden)
        reconstructed = self.output_projection(hidden)
        return reconstructed.transpose(1, 2)


class _ConvAutoencoder(nn.Module):
    """Autoencoder with CNN-like convolutional encoder and a convolutional decoder."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        self.encoder = _ConvAutoencoderEncoder(input_dim=int(input_dim), config=config)
        self.decoder = _ConvAutoencoderDecoder(
            embedding_dim=int(self.encoder.embedding_dim),
            latent_channels=int(config.conv_channels),
            output_dim=int(input_dim),
            config=config,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(inputs, layer_mask)
        reconstructed = self.decoder(embedding, target_length=int(inputs.shape[1]))
        return reconstructed, embedding

    def encode(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs, layer_mask)


class AutoencoderKNNSupervisedModel:
    """
    Autoencoder + KNN classifier model following CNN1DSupervisedModel interface.
    
    The model first trains an autoencoder unsupervised on the feature data,
    then uses the encoder to extract features, and finally trains a KNN classifier
    on top of the extracted features.
    """

    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        n_neighbors: int = 5,
        dropout: float = 0.1,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        random_state: int = 42,
        device: str | None = None,
        pretrained_checkpoint: Path | str | None = None,
        no_train: bool = False,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
    ) -> None:
        _require_torch()

        if int(conv_channels) <= 0:
            raise ValueError("autoencoder_knn conv_channels must be positive")
        if int(num_conv_layers) <= 0:
            raise ValueError("autoencoder_knn num_conv_layers must be positive")
        if int(kernel_size) <= 0:
            raise ValueError("autoencoder_knn kernel_size must be positive")
        if int(stride) <= 0:
            raise ValueError("autoencoder_knn stride must be positive")
        if int(dilation) <= 0:
            raise ValueError("autoencoder_knn dilation must be positive")
        if int(stride) != 1:
            raise ValueError("autoencoder_knn currently requires stride=1 to preserve reconstruction length")
        if n_neighbors <= 0:
            raise ValueError("autoencoder_knn n_neighbors must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("autoencoder_knn dropout must be in [0, 1)")
        if bool(class_weight_loss) and bool(rank_label_weight_loss):
            raise ValueError(
                "autoencoder_knn supports either class_weight_loss or rank_label_weight_loss, not both"
            )
        if bool(no_train) and pretrained_checkpoint is None:
            raise ValueError("autoencoder_knn no_train requires pretrained_checkpoint")

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
        self.n_neighbors = int(n_neighbors)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.normalization = str(normalization)
        self.pooling = str(pooling)
        self.include_total_layer_count = bool(include_total_layer_count)
        self.depth_feature_mode = str(depth_feature_mode)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.device_ = _resolve_torch_device(device)
        self.device_name_ = str(self.device_)
        self.pretrained_checkpoint = (
            None
            if pretrained_checkpoint is None
            else str(Path(pretrained_checkpoint).expanduser().resolve())
        )
        self.no_train = bool(no_train)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight_loss = bool(class_weight_loss)
        self.rank_label_weight_loss = bool(rank_label_weight_loss)
        
        self.autoencoder_: _ConvAutoencoder | None = None
        self.knn_classifier_: KNeighborsClassifier | None = None
        self.normalization_stats_: CNNNormalizationStats | None = None
        self.channel_layout_: CNNChannelLayout | None = None
        self.channel_mean_: np.ndarray | None = None
        self.channel_std_: np.ndarray | None = None
        self.input_channels_: int | None = None
        self.feature_dim_: int | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "autoencoder_knn"
        self._fit_summary: dict[str, Any] = {}
        self._pretrained_checkpoint_payload: dict[str, Any] | None = None
        self._pretrained_autoencoder_state_dict: dict[str, Any] | None = None
        
        # Store encoder training history
        self._encoder_history: list[dict[str, float]] = []

    def _set_random_seeds(self) -> None:
        assert torch is not None
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _set_threads(self, n_jobs: int | None) -> None:
        assert torch is not None
        if n_jobs is None:
            return
        resolved = int(n_jobs)
        if resolved <= 0:
            return
        torch.set_num_threads(resolved)

    def _prepare_numpy_inputs(self, bundle: SupervisedFeatureBundle) -> CNNFeatureTensors:
        """Prepare bundle into tensor format, similar to CNN1DSupervisedModel."""
        vector_batch = build_per_layer_vectors(
            bundle,
            normalization_stats=self.normalization_stats_,
            include_total_layer_count=bool(self.include_total_layer_count),
            depth_feature_mode=str(self.depth_feature_mode),
        )
        if self.channel_layout_ is None:
            self.channel_layout_ = vector_batch.channel_layout
        elif self.channel_layout_ != vector_batch.channel_layout:
            raise ValueError(
                "autoencoder_knn encountered an incompatible layer-sequence channel layout between bundles"
            )

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
        elif (
            self.channel_mean_ is None
            or self.channel_std_ is None
        ):
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

    def _build_autoencoder(self, *, input_channels: int) -> _ConvAutoencoder:
        """Build the autoencoder model."""
        _require_torch()
        autoencoder = _ConvAutoencoder(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
        )
        autoencoder = autoencoder.to(self.device_)
        self.input_channels_ = int(input_channels)
        return autoencoder

    def _load_pretrained_checkpoint_metadata(self) -> None:
        if self.pretrained_checkpoint is None:
            return

        assert torch is not None
        from .cnn import _channel_layout_from_payload, _torch_load_checkpoint

        payload = _torch_load_checkpoint(Path(self.pretrained_checkpoint))
        backend = str(payload.get("backend") or "autoencoder_knn")
        if backend != "autoencoder_knn":
            raise ValueError(f"Unsupported autoencoder_knn checkpoint backend={backend!r}")

        channel_layout_payload = payload.get("channel_layout")
        normalization_payload = payload.get("normalization")
        if not isinstance(channel_layout_payload, dict):
            raise ValueError("autoencoder_knn pretrained checkpoint is missing channel_layout")
        if not isinstance(normalization_payload, dict):
            raise ValueError("autoencoder_knn pretrained checkpoint is missing normalization")

        channel_layout = _channel_layout_from_payload(channel_layout_payload)
        channel_mean = np.asarray(normalization_payload.get("channel_mean"), dtype=np.float32)
        channel_std = np.asarray(normalization_payload.get("channel_std"), dtype=np.float32)
        if channel_mean.ndim != 1 or channel_std.ndim != 1 or channel_mean.shape != channel_std.shape:
            raise ValueError("autoencoder_knn pretrained checkpoint normalization arrays must be aligned 1D arrays")

        autoencoder_state_dict = payload.get("autoencoder_state_dict")
        if not isinstance(autoencoder_state_dict, dict):
            raise ValueError("autoencoder_knn pretrained checkpoint is missing autoencoder_state_dict")

        config = payload.get("config")
        if isinstance(config, dict) and config.get("input_channels") is not None:
            self.input_channels_ = int(config.get("input_channels"))
        else:
            self.input_channels_ = int(channel_layout.input_dim)

        self._pretrained_checkpoint_payload = payload
        self._pretrained_autoencoder_state_dict = autoencoder_state_dict
        self.channel_layout_ = channel_layout
        self.normalization_stats_ = CNNNormalizationStats(
            channel_mean=channel_mean,
            channel_std=channel_std,
        )
        self.channel_mean_ = channel_mean
        self.channel_std_ = channel_std

    def _move_batch_to_device(self, batch_inputs: Any) -> Any:
        assert torch is not None
        return batch_inputs.to(self.device_, non_blocking=False)

    def _extract_features_from_loader(self, loader: DataLoader) -> np.ndarray:
        """Extract embeddings from autoencoder."""
        assert torch is not None
        if self.autoencoder_ is None:
            raise RuntimeError("autoencoder_knn autoencoder has not been fit")
        
        self.autoencoder_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                batch_inputs = self._move_batch_to_device(batch_inputs)
                layer_mask_device = batch_layer_mask.to(self.device_)
                flattened_embeddings = self.autoencoder_.encode(batch_inputs, layer_mask_device)
                if self.feature_dim_ is None:
                    self.feature_dim_ = int(flattened_embeddings.shape[1])
                outputs.append(flattened_embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
        
        if not outputs:
            return np.asarray([], dtype=np.float32).reshape(0, int(self.feature_dim_ or 0))
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        rank_labels: np.ndarray | None = None,
    ) -> "AutoencoderKNNSupervisedModel":
        """
        Fit the autoencoder (unsupervised) then train KNN on extracted features.
        """
        _require_torch()
        assert torch is not None
        
        self._set_random_seeds()
        self._set_threads(n_jobs)

        if self.pretrained_checkpoint is not None:
            self._load_pretrained_checkpoint_metadata()

        train_tensors = self._prepare_numpy_inputs(bundle)
        y_train = np.asarray(labels, dtype=np.int64).reshape(-1)
        
        if train_tensors.inputs.shape[0] != y_train.shape[0]:
            raise ValueError("autoencoder_knn training features/labels length mismatch")

        # ============================================================
        # Phase 1: Train autoencoder unsupervised
        # ============================================================
        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
        )

        self.autoencoder_ = self._build_autoencoder(
            input_channels=int(train_tensors.inputs.shape[2])
        )
        if self._pretrained_autoencoder_state_dict is not None:
            self.autoencoder_.load_state_dict(self._pretrained_autoencoder_state_dict)
        
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self.autoencoder_.state_dict().items()
        }
        best_metric: float | None = None
        best_epoch = -1
        self._encoder_history = []

        if not self.no_train:
            optimizer = torch.optim.AdamW(
                self.autoencoder_.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            best_metric_value = math.inf
            stale_epochs = 0

            for epoch_idx in range(self.max_epochs):
                self.autoencoder_.train()
                epoch_loss = 0.0
                batch_count = 0

                for batch_inputs, batch_layer_mask in train_loader:
                    batch_inputs = self._move_batch_to_device(batch_inputs)
                    batch_layer_mask = batch_layer_mask.to(self.device_)
                    optimizer.zero_grad()
                    reconstructed, _ = self.autoencoder_(batch_inputs, batch_layer_mask)
                    valid_mask = batch_layer_mask.unsqueeze(-1).to(dtype=batch_inputs.dtype)
                    squared_error = torch.square(reconstructed - batch_inputs) * valid_mask
                    loss = torch.sum(squared_error) / torch.clamp(torch.sum(valid_mask), min=1.0)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.detach().cpu().item())
                    batch_count += 1

                avg_loss = epoch_loss / max(1, batch_count)

                # Validation on reconstruction loss
                if validation_data is not None:
                    valid_bundle, valid_labels = validation_data
                    valid_tensors = self._prepare_numpy_inputs(valid_bundle)
                    valid_dataset = TensorDataset(
                        torch.from_numpy(valid_tensors.inputs),
                        torch.from_numpy(valid_tensors.layer_mask),
                    )
                    valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                        shuffle=False,
                    )

                    self.autoencoder_.eval()
                    valid_loss = 0.0
                    valid_count = 0
                    with torch.no_grad():
                        for batch_inputs, batch_layer_mask in valid_loader:
                            batch_inputs = self._move_batch_to_device(batch_inputs)
                            batch_layer_mask = batch_layer_mask.to(self.device_)
                            reconstructed, _ = self.autoencoder_(batch_inputs, batch_layer_mask)
                            valid_mask = batch_layer_mask.unsqueeze(-1).to(dtype=batch_inputs.dtype)
                            squared_error = torch.square(reconstructed - batch_inputs) * valid_mask
                            loss = torch.sum(squared_error) / torch.clamp(torch.sum(valid_mask), min=1.0)
                            valid_loss += float(loss.detach().cpu().item())
                            valid_count += 1

                    metric = valid_loss / max(1, valid_count)
                else:
                    metric = avg_loss

                self._encoder_history.append({
                    "epoch": float(epoch_idx),
                    "train_loss": float(avg_loss),
                    "selection_metric": float(metric),
                    "selection_metric_name": "reconstruction_mse",
                })

                if metric < best_metric_value:
                    best_metric_value = float(metric)
                    best_epoch = int(epoch_idx)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.autoencoder_.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if validation_data is not None and stale_epochs >= self.patience:
                    break

            best_metric = float(best_metric_value)
        else:
            self.autoencoder_.eval()

        self.autoencoder_.load_state_dict(best_state)
        
        # ============================================================
        # Phase 2: Extract features using trained autoencoder
        # ============================================================
        train_loader_inference = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=False,
        )
        
        train_features = self._extract_features_from_loader(train_loader_inference)
        self.feature_dim_ = int(train_features.shape[1]) if train_features.ndim == 2 else None
        
        # ============================================================
        # Phase 3: Train KNN classifier on extracted features
        # ============================================================
        self.knn_classifier_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=n_jobs,
        )
        self.knn_classifier_.fit(train_features, y_train)
        
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": None if best_metric is None else float(best_metric),
            "selection_metric_name": "pretrained_checkpoint" if self.no_train else "reconstruction_mse",
            "epochs_ran": int(len(self._encoder_history)),
            "history": self._encoder_history,
            "feature_dim": int(self.feature_dim_ or 0),
            "conv_channels": int(self.conv_channels),
            "n_neighbors": int(self.n_neighbors),
            "class_weight_loss": bool(self.class_weight_loss),
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "training_mode": "pretrained_frozen"
            if self.no_train
            else ("pretrained_finetune" if self._pretrained_autoencoder_state_dict is not None else "scratch"),
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "no_train": bool(self.no_train),
        }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        """Get decision scores for binary classification."""
        _require_torch()
        assert torch is not None
        if self.knn_classifier_ is None:
            raise RuntimeError("autoencoder_knn model has not been fit")
        
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        
        features = self._extract_features_from_loader(loader)
        
        if self.task_spec.is_binary:
            # For binary, return scores (probability of positive class - probability of negative class)
            probabilities = self.knn_classifier_.predict_proba(features)
            return (probabilities[:, 1] - probabilities[:, 0]).astype(np.float64, copy=False)
        else:
            # For multiclass, return logits approximated from probabilities
            probabilities = self.knn_classifier_.predict_proba(features)
            # Use log-odds as a proxy for logits
            logits = np.log(probabilities + 1e-10)
            return logits.astype(np.float64, copy=False)

    def extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        """Extract features using the encoder."""
        _require_torch()
        assert torch is not None
        if self.autoencoder_ is None:
            raise RuntimeError("autoencoder_knn model has not been fit")
        
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        return self._extract_features_from_loader(loader)

    def predict_proba(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        """Get class probabilities."""
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        
        features = self._extract_features_from_loader(loader)
        
        if self.knn_classifier_ is None:
            raise RuntimeError("autoencoder_knn model has not been fit")
        
        return self.knn_classifier_.predict_proba(features).astype(np.float64, copy=False)

    def predict(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        """Get predicted class labels."""
        if self.knn_classifier_ is None:
            raise RuntimeError("autoencoder_knn model has not been fit")
        
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        
        features = self._extract_features_from_loader(loader)
        return self.knn_classifier_.predict(features).astype(np.int32, copy=False)

    def reconstruction_losses(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        """Compute per-sample reconstruction loss (MSE over valid elements) for a bundle.

        Returns a 1-D float64 numpy array with one loss per sample in the same order
        as the rows that would be written by `save_score_csv` (i.e., bundle order).
        """
        _require_torch()
        assert torch is not None
        if self.autoencoder_ is None:
            raise RuntimeError("autoencoder_knn autoencoder has not been fit")

        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )

        self.autoencoder_.eval()
        losses: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                batch_inputs = self._move_batch_to_device(batch_inputs)
                layer_mask_device = batch_layer_mask.to(self.device_)
                reconstructed, _ = self.autoencoder_(batch_inputs, layer_mask_device)
                valid_mask = batch_layer_mask.unsqueeze(-1).to(dtype=batch_inputs.dtype)
                squared_error = torch.square(reconstructed - batch_inputs) * valid_mask
                per_sample_sum = torch.sum(squared_error, dim=(1, 2))
                per_sample_count = torch.clamp(torch.sum(valid_mask, dim=(1, 2)), min=1.0)
                per_sample_loss = (per_sample_sum / per_sample_count).detach().cpu().numpy().astype(
                    np.float64
                )
                losses.append(per_sample_loss)

        if not losses:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(losses, axis=0).astype(np.float64, copy=False)

    @property
    def fit_summary(self) -> dict[str, Any]:
        """Get the fit summary containing training details and history."""
        if not self._fit_summary:
            return {}
        return dict(self._fit_summary)

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        """Extra data to save in checkpoint."""
        return {
            "autoencoder_architecture": {
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
            }
        }

    def save(self, path: Path) -> None:
        """Save the model to disk."""
        _require_torch()
        assert torch is not None
        if (
            self.autoencoder_ is None
            or self.knn_classifier_ is None
            or self.channel_mean_ is None
            or self.channel_std_ is None
            or self.channel_layout_ is None
        ):
            raise RuntimeError("autoencoder_knn model has not been fit")
        
        payload = {
            "backend": self.backend_name_,
            "config": {
                "conv_channels": int(self.conv_channels),
                "num_conv_layers": int(self.num_conv_layers),
                "kernel_size": int(self.kernel_size),
                "stride": int(self.stride),
                "dilation": int(self.dilation),
                "n_neighbors": int(self.n_neighbors),
                "dropout": float(self.dropout),
                "use_residual": bool(self.use_residual),
                "normalization": str(self.normalization),
                "pooling": str(self.pooling),
                "include_total_layer_count": bool(self.include_total_layer_count),
                "depth_feature_mode": str(self.depth_feature_mode),
                "device": str(self.device_name_),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
                "random_state": int(self.random_state),
                "pretrained_checkpoint": self.pretrained_checkpoint,
                "no_train": bool(self.no_train),
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "patience": int(self.patience),
                "class_weight_loss": bool(self.class_weight_loss),
                "rank_label_weight_loss": bool(self.rank_label_weight_loss),
                "input_channels": int(self.input_channels_ or 0),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "channel_layout": asdict(self.channel_layout_),
            "autoencoder_state_dict": self.autoencoder_.state_dict(),
            "knn_classifier": self.knn_classifier_,
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


def load_autoencoder_knn_checkpoint(
    path: Path,
    device: str | None = None,
) -> AutoencoderKNNSupervisedModel:
    """Load a saved autoencoder+KNN model."""
    _require_torch()
    assert torch is not None
    
    from .cnn import _torch_load_checkpoint, _channel_layout_from_payload
    
    payload = _torch_load_checkpoint(path)
    backend = str(payload.get("backend") or "autoencoder_knn")
    if backend != "autoencoder_knn":
        raise ValueError(f"Unsupported autoencoder_knn checkpoint backend={backend!r}")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("autoencoder_knn checkpoint is missing config")
    
    task_spec = _task_spec_from_payload(payload.get("task"))
    checkpoint_device = str(config.get("device", "cpu"))
    chosen_device = str(device) if device is not None else checkpoint_device
    if device is None and chosen_device.startswith("cuda") and not torch.cuda.is_available():
        chosen_device = "cpu"

    model = AutoencoderKNNSupervisedModel(
        conv_channels=int(config.get("conv_channels", 64)),
        num_conv_layers=int(config.get("num_conv_layers", 3)),
        kernel_size=int(config.get("kernel_size", 3)),
        stride=int(config.get("stride", 1)),
        dilation=int(config.get("dilation", 1)),
        n_neighbors=int(config["n_neighbors"]),
        dropout=float(config.get("dropout", 0.1)),
        use_residual=bool(config.get("use_residual", True)),
        normalization=str(config.get("normalization", "layernorm")),
        pooling=str(config.get("pooling", "mean_max")),
        include_total_layer_count=bool(config.get("include_total_layer_count", True)),
        depth_feature_mode=str(config.get("depth_feature_mode", "both")),
        device=chosen_device,
        pretrained_checkpoint=config.get("pretrained_checkpoint"),
        no_train=bool(config.get("no_train", False)),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        random_state=int(config.get("random_state", 42)),
        task_spec=task_spec,
        max_epochs=int(config.get("max_epochs", CNN_MAX_EPOCHS)),
        batch_size=int(config.get("batch_size", CNN_BATCH_SIZE)),
        patience=int(config.get("patience", CNN_PATIENCE)),
        class_weight_loss=bool(config.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(config.get("rank_label_weight_loss", False)),
    )

    channel_layout = _channel_layout_from_payload(payload.get("channel_layout"))
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, dict):
        raise ValueError("autoencoder_knn checkpoint is missing normalization")
    
    channel_mean = np.asarray(normalization_payload.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(normalization_payload.get("channel_std"), dtype=np.float32)
    if channel_mean.ndim != 1 or channel_std.ndim != 1 or channel_mean.shape != channel_std.shape:
        raise ValueError("autoencoder_knn checkpoint normalization arrays must be aligned 1D arrays")

    model.channel_layout_ = channel_layout
    model.normalization_stats_ = CNNNormalizationStats(
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    model.channel_mean_ = channel_mean
    model.channel_std_ = channel_std
    model.input_channels_ = int(config.get("input_channels") or channel_layout.input_dim)
    model.classes_ = np.asarray(payload.get("classes", np.arange(task_spec.n_classes)), dtype=np.int32)
    model.class_names_ = tuple(str(x) for x in payload.get("class_names", task_spec.class_names))
    fit_summary = payload.get("fit_summary")
    model._fit_summary = dict(fit_summary) if isinstance(fit_summary, dict) else {}

    # Rebuild and load autoencoder
    model.autoencoder_ = model._build_autoencoder(
        input_channels=int(model.input_channels_ or channel_layout.input_dim)
    )
    autoencoder_state_dict = payload.get("autoencoder_state_dict")
    if not isinstance(autoencoder_state_dict, dict):
        raise ValueError("autoencoder_knn checkpoint is missing autoencoder_state_dict")
    model.autoencoder_.load_state_dict(autoencoder_state_dict)
    model.autoencoder_ = model.autoencoder_.to(model.device_)
    model.autoencoder_.eval()
    
    # Load KNN classifier
    model.knn_classifier_ = payload.get("knn_classifier")
    if model.knn_classifier_ is None:
        raise ValueError("autoencoder_knn checkpoint is missing knn_classifier")

    return model