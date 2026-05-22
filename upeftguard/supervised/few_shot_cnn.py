from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .interfaces import SupervisedFeatureBundle, SupervisedTaskSpec
from .cnn import (
    CNN_BATCH_SIZE,
    CNNChannelLayout,
    CNNNormalizationStats,
    _default_binary_task_spec,
    _require_torch,
    load_cnn_checkpoint,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
except Exception:  # pragma: no cover - soft dependency
    torch = None


class FewShotCNNSupervisedModel:
    """Few-shot KNN classifier on top of a frozen CNN feature extractor."""

    def __init__(
        self,
        *,
        k_knn: int,
        n_few_shot_per_class: int,
        pretrained_checkpoint: Path | str | None,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        batch_size: int = CNN_BATCH_SIZE,
    ) -> None:
        _require_torch()
        if int(k_knn) <= 0:
            raise ValueError("few_shot_cnn k_knn must be positive")
        if int(n_few_shot_per_class) <= 0:
            raise ValueError("few_shot_cnn n_few_shot_per_class must be positive")
        if pretrained_checkpoint is None or str(pretrained_checkpoint).strip() == "":
            raise ValueError("few_shot_cnn requires pretrained_checkpoint")

        self.k_knn = int(k_knn)
        self.n_few_shot_per_class = int(n_few_shot_per_class)
        self.pretrained_checkpoint = str(Path(pretrained_checkpoint).expanduser().resolve())
        self.random_state = int(random_state)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.batch_size = int(batch_size)

        self.cnn_: Any | None = None
        self.knn_classifier_: KNeighborsClassifier | None = None
        self.normalization_stats_: CNNNormalizationStats | None = None
        self.channel_layout_: CNNChannelLayout | None = None
        self.channel_mean_: np.ndarray | None = None
        self.channel_std_: np.ndarray | None = None
        self.input_channels_: int | None = None
        self.feature_dim_: int | None = None
        self.few_shot_indices_: np.ndarray | None = None
        self.few_shot_labels_: np.ndarray | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "few_shot_cnn"
        self._fit_summary: dict[str, Any] = {}
        self._cnn_config: dict[str, Any] | None = None
        self._cnn_backend: str | None = None

    def _ensure_cnn_loaded(self) -> None:
        if self.cnn_ is not None:
            return
        if not self.pretrained_checkpoint:
            raise ValueError("few_shot_cnn requires pretrained_checkpoint")
        checkpoint_path = Path(self.pretrained_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"few_shot_cnn checkpoint not found: {checkpoint_path}")

        self.cnn_ = load_cnn_checkpoint(checkpoint_path)
        self.cnn_.batch_size = int(self.batch_size)
        self._cnn_backend = str(getattr(self.cnn_, "backend_name_", "cnn_1d"))
        self._cnn_config = {
            "conv_channels": int(self.cnn_.conv_channels),
            "num_conv_layers": int(self.cnn_.num_conv_layers),
            "kernel_size": int(self.cnn_.kernel_size),
            "stride": int(self.cnn_.stride),
            "dilation": int(self.cnn_.dilation),
            "dropout": float(self.cnn_.dropout),
            "use_residual": bool(self.cnn_.use_residual),
            "normalization": str(self.cnn_.normalization),
            "pooling": str(self.cnn_.pooling),
            "include_total_layer_count": bool(self.cnn_.include_total_layer_count),
            "depth_feature_mode": str(self.cnn_.depth_feature_mode),
            "learning_rate": float(self.cnn_.learning_rate),
            "weight_decay": float(self.cnn_.weight_decay),
            "random_state": int(self.cnn_.random_state),
            "max_epochs": int(self.cnn_.max_epochs),
            "batch_size": int(self.cnn_.batch_size),
            "patience": int(self.cnn_.patience),
            "class_weight_loss": bool(self.cnn_.class_weight_loss),
            "rank_label_weight_loss": bool(self.cnn_.rank_label_weight_loss),
            "input_channels": int(self.cnn_.input_channels_ or 0),
            "task_mode": str(self.cnn_.task_spec.task_mode),
            "num_classes": int(self.cnn_.task_spec.n_classes),
        }

        self.channel_layout_ = self.cnn_.channel_layout_
        self.normalization_stats_ = self.cnn_.normalization_stats_
        self.channel_mean_ = self.cnn_.channel_mean_
        self.channel_std_ = self.cnn_.channel_std_
        self.input_channels_ = self.cnn_.input_channels_

    def _validate_bundle(self, bundle: SupervisedFeatureBundle) -> SupervisedFeatureBundle:
        if not isinstance(bundle, SupervisedFeatureBundle):
            raise ValueError("few_shot_cnn requires a structured feature bundle")
        return bundle

    def _validate_labels(self, labels: np.ndarray) -> np.ndarray:
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_np.size == 0:
            raise ValueError("few_shot_cnn requires at least one labeled sample")
        if np.any(labels_np < 0) or np.any(labels_np >= int(self.task_spec.n_classes)):
            raise ValueError("few_shot_cnn labels must be within configured task classes")
        return labels_np

    def _select_few_shot_indices(
        self,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, int]]:
        rng = np.random.default_rng(int(self.random_state))
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        selected_indices: list[int] = []
        per_class_counts: dict[int, int] = {}
        for class_idx in range(int(self.task_spec.n_classes)):
            class_indices = np.flatnonzero(labels_np == int(class_idx))
            if class_indices.size == 0:
                per_class_counts[int(class_idx)] = 0
                continue
            rng.shuffle(class_indices)
            take = min(int(self.n_few_shot_per_class), int(class_indices.size))
            per_class_counts[int(class_idx)] = int(take)
            selected_indices.extend(class_indices[:take].tolist())
        if not selected_indices:
            raise ValueError("few_shot_cnn could not select any few-shot samples")
        return np.asarray(selected_indices, dtype=np.int64), per_class_counts

    def _extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        self._ensure_cnn_loaded()
        assert self.cnn_ is not None
        return np.asarray(self.cnn_.extract_features(bundle), dtype=np.float32)

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        rank_labels: np.ndarray | None = None,
    ) -> "FewShotCNNSupervisedModel":
        _require_torch()
        _ = validation_data
        _ = rank_labels

        bundle = self._validate_bundle(bundle)
        labels_np = self._validate_labels(labels)
        if int(bundle.n_samples) != int(labels_np.shape[0]):
            raise ValueError("few_shot_cnn training features/labels length mismatch")

        selected_indices, per_class_counts = self._select_few_shot_indices(labels_np)
        if int(self.k_knn) > int(selected_indices.size):
            raise ValueError(
                "few_shot_cnn k_knn must be <= number of selected few-shot samples "
                f"({self.k_knn} > {selected_indices.size})"
            )

        few_shot_bundle = bundle.subset(selected_indices)
        few_shot_labels = labels_np[selected_indices]
        features = self._extract_features(few_shot_bundle)
        if features.ndim != 2 or int(features.shape[0]) != int(few_shot_labels.shape[0]):
            raise ValueError("few_shot_cnn feature extraction produced an unexpected shape")

        self.knn_classifier_ = KNeighborsClassifier(
            n_neighbors=int(self.k_knn),
            n_jobs=n_jobs,
        )
        self.knn_classifier_.fit(features, few_shot_labels)
        self.feature_dim_ = int(features.shape[1])
        self.few_shot_indices_ = np.asarray(selected_indices, dtype=np.int64)
        self.few_shot_labels_ = np.asarray(few_shot_labels, dtype=np.int64)
        self.classes_ = np.asarray(self.knn_classifier_.classes_, dtype=np.int32)

        self._fit_summary = {
            "few_shot_total": int(selected_indices.size),
            "few_shot_per_class": {str(k): int(v) for k, v in per_class_counts.items()},
            "feature_dim": int(self.feature_dim_),
            "k_knn": int(self.k_knn),
            "n_few_shot_per_class": int(self.n_few_shot_per_class),
            "pretrained_checkpoint": str(self.pretrained_checkpoint),
            "cnn_backend": str(self._cnn_backend or ""),
        }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        if self.knn_classifier_ is None:
            raise RuntimeError("few_shot_cnn model has not been fit")
        probabilities = self.predict_proba(bundle)
        if self.task_spec.is_binary:
            return (probabilities[:, 1] - probabilities[:, 0]).astype(np.float64, copy=False)
        return np.log(probabilities + 1e-10).astype(np.float64, copy=False)

    def extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        bundle = self._validate_bundle(bundle)
        return self._extract_features(bundle)

    def predict_proba(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        if self.knn_classifier_ is None:
            raise RuntimeError("few_shot_cnn model has not been fit")
        bundle = self._validate_bundle(bundle)
        features = self._extract_features(bundle)
        return self.knn_classifier_.predict_proba(features).astype(np.float64, copy=False)

    def predict(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        if self.knn_classifier_ is None:
            raise RuntimeError("few_shot_cnn model has not been fit")
        bundle = self._validate_bundle(bundle)
        features = self._extract_features(bundle)
        return self.knn_classifier_.predict(features).astype(np.int32, copy=False)

    @property
    def fit_summary(self) -> dict[str, Any]:
        if not self._fit_summary:
            return {}
        return dict(self._fit_summary)

    def save(self, path: Path) -> None:
        _require_torch()
        assert torch is not None
        if (
            self.cnn_ is None
            or self.knn_classifier_ is None
            or self.channel_layout_ is None
            or self.channel_mean_ is None
            or self.channel_std_ is None
        ):
            raise RuntimeError("few_shot_cnn model has not been fit")

        payload = {
            "backend": self.backend_name_,
            "config": {
                "k_knn": int(self.k_knn),
                "n_few_shot_per_class": int(self.n_few_shot_per_class),
                "random_state": int(self.random_state),
                "pretrained_checkpoint": str(self.pretrained_checkpoint),
                "input_channels": int(self.input_channels_ or 0),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "cnn": {
                "backend": str(self._cnn_backend or "cnn_1d"),
                "config": dict(self._cnn_config or {}),
                "channel_layout": asdict(self.channel_layout_),
                "state_dict": self.cnn_.model_.state_dict(),
                "normalization": {
                    "channel_mean": np.asarray(self.channel_mean_, dtype=np.float32),
                    "channel_std": np.asarray(self.channel_std_, dtype=np.float32),
                },
            },
            "knn_classifier": self.knn_classifier_,
            "classes": np.asarray(self.classes_, dtype=np.int32),
            "class_names": list(self.class_names_),
            "task": self.task_spec.to_dict(),
            "fit_summary": dict(self._fit_summary),
        }
        torch.save(payload, Path(path).expanduser().resolve())
