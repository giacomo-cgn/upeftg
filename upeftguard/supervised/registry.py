from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .interfaces import (
    ARCHITECTURE_INDEPENDENT_AGGREGATION_KIND,
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    SupervisedTaskSpec,
    TABULAR_SPECTRAL_REPRESENTATION_KIND,
)


NormalizationFactory = Callable[[], Any]
EstimatorFactory = Callable[[dict[str, Any], int, SupervisedTaskSpec | None], Any]


TABULAR_REPRESENTATION_KINDS = (
    TABULAR_SPECTRAL_REPRESENTATION_KIND,
    ARCHITECTURE_INDEPENDENT_AGGREGATION_KIND,
)
CNN_1D_HYPERPARAM_NAMES = (
    "conv_channels",
    "num_conv_layers",
    "kernel_size",
    "dropout",
    "learning_rate",
    "weight_decay",
)
CNN_1D_MODEL_NAME = "cnn_1d"
CNN_1D_DANN_MODEL_NAME = "cnn_1d_dann"
AUTOENCODER_KNN_MODEL_NAME = "autoencoder_knn"
AUTOENCODER_KNN_HYPERPARAM_NAMES = (
    "conv_channels",
    "num_conv_layers",
    "kernel_size",
    "n_neighbors",
    "knn",
    "dropout",
    "learning_rate",
    "weight_decay",
)
AUTOENCODER_KNN_PRETRAINED_HYPERPARAM_NAMES = AUTOENCODER_KNN_HYPERPARAM_NAMES + (
    "pretrained_checkpoint",
    "no_train",
)
FEW_SHOT_CNN_MODEL_NAME = "few_shot_cnn"
FEW_SHOT_CNN_HYPERPARAM_NAMES = (
    "k_knn",
    "n_few_shot_per_class",
)
_CNN_1D_INTEGER_HYPERPARAMS = {"conv_channels", "num_conv_layers", "kernel_size"}
_CNN_1D_FLOAT_HYPERPARAMS = {"dropout", "learning_rate", "weight_decay"}


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    backend: str
    complexity_rank: int
    normalization_policy: str
    normalization_factory: NormalizationFactory
    param_grid: tuple[dict[str, Any], ...]
    estimator_factory: EstimatorFactory
    supported_representation_kinds: tuple[str, ...]


def _grid(**axes: list[Any]) -> tuple[dict[str, Any], ...]:
    keys = list(axes.keys())
    values = [list(axes[key]) for key in keys]
    return tuple(
        {key: value for key, value in zip(keys, combo)}
        for combo in product(*values)
    )


def _passthrough() -> str:
    return "passthrough"


def _standard_scaler() -> StandardScaler:
    return StandardScaler()


def default_cnn_hyperparams_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "cnn_hyperparams" / "cnn_1d_default.json"


def default_autoencoder_hyperparams_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "selfsupervised_hyperparams" / "autoencoder_hyperparams.json"


def default_few_shot_hyperparams_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "few_shot" / "few_shot_knn.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CNN hyperparameter JSON not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"CNN hyperparameter JSON must be an object at the top level, got {type(payload).__name__}"
        )
    return payload


def _normalize_cnn_hyperparam_axes(payload: dict[str, Any]) -> dict[str, list[Any]]:
    raw_keys = {str(key) for key in payload.keys()}
    expected_keys = set(CNN_1D_HYPERPARAM_NAMES)
    missing = sorted(expected_keys - raw_keys)
    extra = sorted(raw_keys - expected_keys)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "CNN hyperparameter JSON must define exactly these keys: "
            f"{list(CNN_1D_HYPERPARAM_NAMES)} ({'; '.join(details)})"
        )

    normalized: dict[str, list[Any]] = {}
    for name in CNN_1D_HYPERPARAM_NAMES:
        raw_values = payload.get(name)
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                f"CNN hyperparameter '{name}' must be a non-empty JSON list, got {type(raw_values).__name__}"
            )
        if name in _CNN_1D_INTEGER_HYPERPARAMS:
            normalized[name] = [int(value) for value in raw_values]
        elif name in _CNN_1D_FLOAT_HYPERPARAMS:
            normalized[name] = [float(value) for value in raw_values]
        else:
            raise ValueError(f"Unsupported CNN hyperparameter axis {name!r}")
    return normalized


def resolve_cnn_hyperparams(
    cnn_hyperparams: Path | str | dict[str, Any] | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if cnn_hyperparams is None:
        source_path = default_cnn_hyperparams_path()
        payload = _load_json_object(source_path)
        source = {
            "source": "default_file",
            "path": str(source_path),
        }
    elif isinstance(cnn_hyperparams, dict):
        payload = dict(cnn_hyperparams)
        source = {
            "source": "inline_object",
            "path": None,
        }
    else:
        source_path = Path(cnn_hyperparams).expanduser().resolve()
        payload = _load_json_object(source_path)
        source = {
            "source": "file",
            "path": str(source_path),
        }

    axes = _normalize_cnn_hyperparam_axes(payload)
    metadata = {
        **source,
        "axes": {name: list(values) for name, values in axes.items()},
        "n_candidates": int(len(_grid(**axes))),
    }
    return axes, metadata


def _normalize_autoencoder_hyperparam_axes(payload: dict[str, Any]) -> dict[str, list[Any]]:
    raw_keys = {str(key) for key in payload.keys()}
    base_expected_keys = set(AUTOENCODER_KNN_HYPERPARAM_NAMES)
    pretrained_expected_keys = set(AUTOENCODER_KNN_PRETRAINED_HYPERPARAM_NAMES)
    legacy_base_keys = base_expected_keys - {"knn"}
    legacy_pretrained_keys = pretrained_expected_keys - {"knn"}
    allow_missing_knn = False
    if raw_keys == base_expected_keys:
        expected_keys = AUTOENCODER_KNN_HYPERPARAM_NAMES
    elif raw_keys == pretrained_expected_keys:
        expected_keys = AUTOENCODER_KNN_PRETRAINED_HYPERPARAM_NAMES
    elif raw_keys == legacy_base_keys:
        expected_keys = AUTOENCODER_KNN_HYPERPARAM_NAMES
        allow_missing_knn = True
    elif raw_keys == legacy_pretrained_keys:
        expected_keys = AUTOENCODER_KNN_PRETRAINED_HYPERPARAM_NAMES
        allow_missing_knn = True
    else:
        missing_base = sorted(base_expected_keys - raw_keys)
        missing_pretrained = sorted(pretrained_expected_keys - raw_keys)
        extra = sorted(raw_keys - pretrained_expected_keys)
        details: list[str] = []
        if missing_base:
            details.append(f"missing_base={missing_base}")
        if missing_pretrained and missing_pretrained != missing_base:
            details.append(f"missing_pretrained={missing_pretrained}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "autoencoder hyperparameter JSON must define exactly these keys: "
            f"{list(AUTOENCODER_KNN_HYPERPARAM_NAMES)} or {list(AUTOENCODER_KNN_PRETRAINED_HYPERPARAM_NAMES)}"
            f" ({'; '.join(details)})"
        )

    normalized: dict[str, list[Any]] = {}
    for name in expected_keys:
        raw_values = payload.get(name)
        if raw_values is None and name == "knn" and allow_missing_knn:
            raw_values = [True]
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                f"autoencoder hyperparameter '{name}' must be a non-empty JSON list, got {type(raw_values).__name__}"
            )
        if name in {"conv_channels", "num_conv_layers", "kernel_size", "n_neighbors"}:
            normalized[name] = [int(value) for value in raw_values]
        elif name == "knn":
            normalized[name] = [bool(value) for value in raw_values]
        elif name in {"dropout", "learning_rate", "weight_decay"}:
            normalized[name] = [float(value) for value in raw_values]
        elif name == "pretrained_checkpoint":
            normalized[name] = [
                None if value is None or str(value).strip() == "" else str(Path(value).expanduser().resolve())
                for value in raw_values
            ]
        elif name == "no_train":
            normalized[name] = [bool(value) for value in raw_values]
        else:
            raise ValueError(f"Unsupported autoencoder hyperparameter axis {name!r}")
    return normalized


def resolve_autoencoder_hyperparams(
    autoencoder_hyperparams: Path | str | dict[str, Any] | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if autoencoder_hyperparams is None:
        source_path = default_autoencoder_hyperparams_path()
        payload = _load_json_object(source_path)
        source = {
            "source": "default_file",
            "path": str(source_path),
        }
    elif isinstance(autoencoder_hyperparams, dict):
        payload = dict(autoencoder_hyperparams)
        source = {
            "source": "inline_object",
            "path": None,
        }
    else:
        source_path = Path(autoencoder_hyperparams).expanduser().resolve()
        payload = _load_json_object(source_path)
        source = {
            "source": "file",
            "path": str(source_path),
        }

    axes = _normalize_autoencoder_hyperparam_axes(payload)
    metadata = {
        **source,
        "axes": {name: list(values) for name, values in axes.items()},
        "n_candidates": int(len(_grid(**axes))),
    }
    return axes, metadata


def _normalize_few_shot_hyperparam_axes(payload: dict[str, Any]) -> dict[str, list[Any]]:
    raw_keys = {str(key) for key in payload.keys()}
    expected_keys = set(FEW_SHOT_CNN_HYPERPARAM_NAMES)
    optional_keys = {"pretrained_checkpoint"}
    missing = sorted(expected_keys - raw_keys)
    extra = sorted(raw_keys - expected_keys - optional_keys)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "few_shot_cnn hyperparameter JSON must define exactly these keys: "
            f"{list(FEW_SHOT_CNN_HYPERPARAM_NAMES)} ({'; '.join(details)})"
        )

    normalized: dict[str, list[Any]] = {}
    for name in FEW_SHOT_CNN_HYPERPARAM_NAMES:
        raw_values = payload.get(name)
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                f"few_shot_cnn hyperparameter '{name}' must be a non-empty JSON list, got {type(raw_values).__name__}"
            )
        normalized[name] = [int(value) for value in raw_values]

    if "pretrained_checkpoint" in payload:
        raw_values = payload.get("pretrained_checkpoint")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                "few_shot_cnn hyperparameter 'pretrained_checkpoint' must be a non-empty JSON list"
            )
        normalized["pretrained_checkpoint"] = [
            None
            if value is None or str(value).strip() == ""
            else str(Path(value).expanduser().resolve())
            for value in raw_values
        ]

    return normalized


def resolve_few_shot_hyperparams(
    few_shot_hyperparams: Path | str | dict[str, Any] | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if few_shot_hyperparams is None:
        source_path = default_few_shot_hyperparams_path()
        payload = _load_json_object(source_path)
        source = {
            "source": "default_file",
            "path": str(source_path),
        }
    elif isinstance(few_shot_hyperparams, dict):
        payload = dict(few_shot_hyperparams)
        source = {
            "source": "inline_object",
            "path": None,
        }
    else:
        source_path = Path(few_shot_hyperparams).expanduser().resolve()
        payload = _load_json_object(source_path)
        source = {
            "source": "file",
            "path": str(source_path),
        }

    axes = _normalize_few_shot_hyperparam_axes(payload)
    metadata = {
        **source,
        "axes": {name: list(values) for name, values in axes.items()},
        "n_candidates": int(len(_grid(**axes))),
    }
    return axes, metadata


def _build_pipeline(
    definition: ModelDefinition,
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("normalizer", definition.normalization_factory()),
            ("model", definition.estimator_factory(params, random_state, task_spec)),
        ]
    )


def _create_cnn_1d(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .cnn import CNN1DSupervisedModel

    return CNN1DSupervisedModel(
        conv_channels=int(params["conv_channels"]),
        num_conv_layers=int(params["num_conv_layers"]),
        kernel_size=int(params["kernel_size"]),
        stride=1,
        dilation=1,
        dropout=float(params["dropout"]),
        use_residual=True,
        normalization="layernorm",
        pooling="mean_max",
        include_total_layer_count=True,
        depth_feature_mode="both",
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        random_state=int(random_state),
        task_spec=task_spec,
        class_weight_loss=bool(params.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(params.get("rank_label_weight_loss", False)),
    )


def _create_cnn_1d_dann(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .cnn import CNN1DDANNSupervisedModel

    return CNN1DDANNSupervisedModel(
        conv_channels=int(params["conv_channels"]),
        num_conv_layers=int(params["num_conv_layers"]),
        kernel_size=int(params["kernel_size"]),
        stride=1,
        dilation=1,
        dropout=float(params["dropout"]),
        use_residual=True,
        normalization="layernorm",
        pooling="mean_max",
        include_total_layer_count=True,
        depth_feature_mode="both",
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        random_state=int(random_state),
        task_spec=task_spec,
        source_rank=int(params.get("source_rank", 256)),
        dann_lambda_max=float(params.get("dann_lambda_max", 1.0)),
        dann_lambda_gamma=float(params.get("dann_lambda_gamma", 10.0)),
        dann_lr_alpha=float(params.get("dann_lr_alpha", 10.0)),
        dann_lr_beta=float(params.get("dann_lr_beta", 0.75)),
        class_weight_loss=bool(params.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(params.get("rank_label_weight_loss", False)),
    )


def _create_autoencoder_knn(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .autoencoder import AutoencoderKNNSupervisedModel

    return AutoencoderKNNSupervisedModel(
        conv_channels=int(params.get("conv_channels", 64)),
        num_conv_layers=int(params.get("num_conv_layers", 3)),
        kernel_size=int(params.get("kernel_size", 3)),
        stride=1,
        dilation=1,
        n_neighbors=int(params.get("n_neighbors", 5)),
        knn=bool(params.get("knn", True)),
        dropout=float(params.get("dropout", 0.1)),
        use_residual=True,
        normalization="layernorm",
        pooling="mean_max",
        include_total_layer_count=True,
        depth_feature_mode="both",
        learning_rate=float(params.get("learning_rate", 0.001)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        random_state=int(random_state),
        task_spec=task_spec,
        pretrained_checkpoint=params.get("pretrained_checkpoint"),
        no_train=bool(params.get("no_train", False)),
        class_weight_loss=bool(params.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(params.get("rank_label_weight_loss", False)),
    )


def _create_few_shot_cnn(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .few_shot_cnn import FewShotCNNSupervisedModel

    return FewShotCNNSupervisedModel(
        k_knn=int(params.get("k_knn", 5)),
        n_few_shot_per_class=int(params.get("n_few_shot_per_class", 5)),
        pretrained_checkpoint=params.get("pretrained_checkpoint"),
        random_state=int(random_state),
        task_spec=task_spec,
    )


_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        backend="sklearn",
        complexity_rank=0,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            solver="lbfgs",
            max_iter=5000,
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    "ridge_classifier": ModelDefinition(
        name="ridge_classifier",
        backend="sklearn",
        complexity_rank=1,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            alpha=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: RidgeClassifier(
            alpha=float(params["alpha"]),
            class_weight=params["class_weight"],
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    "linear_svm": ModelDefinition(
        name="linear_svm",
        backend="sklearn",
        complexity_rank=2,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: SVC(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            kernel="linear",
            probability=True,
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    "adaboost": ModelDefinition(
        name="adaboost",
        backend="sklearn",
        complexity_rank=3,
        normalization_policy="passthrough",
        normalization_factory=_passthrough,
        param_grid=_grid(
            max_depth=[1, 2],
            n_estimators=[50, 100, 200, 400],
            learning_rate=[0.05, 0.1, 0.5, 1.0],
        ),
        estimator_factory=lambda params, random_state, task_spec: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=int(params["max_depth"]),
                random_state=int(random_state),
            ),
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    "kernel_svm": ModelDefinition(
        name="kernel_svm",
        backend="sklearn",
        complexity_rank=4,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-2, 1e-1, 1.0, 10.0, 100.0],
            gamma=["scale", 1e-2, 1e-1, 1.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            kernel="rbf",
            probability=False,
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    "random_forest": ModelDefinition(
        name="random_forest",
        backend="sklearn",
        complexity_rank=5,
        normalization_policy="passthrough",
        normalization_factory=_passthrough,
        param_grid=_grid(
            n_estimators=[200, 400, 800],
            max_depth=[None, 8, 16],
            min_samples_leaf=[1, 2, 4],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=params["class_weight"],
            max_features="sqrt",
            n_jobs=1,
            random_state=int(random_state),
        ),
        supported_representation_kinds=TABULAR_REPRESENTATION_KINDS,
    ),
    CNN_1D_MODEL_NAME: ModelDefinition(
        name=CNN_1D_MODEL_NAME,
        backend="cnn",
        complexity_rank=6,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_cnn_1d,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
    CNN_1D_DANN_MODEL_NAME: ModelDefinition(
        name=CNN_1D_DANN_MODEL_NAME,
        backend="cnn",
        complexity_rank=7,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_cnn_1d_dann,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
    AUTOENCODER_KNN_MODEL_NAME: ModelDefinition(
        name=AUTOENCODER_KNN_MODEL_NAME,
        backend="autoencoder_knn",
        complexity_rank=8,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_autoencoder_knn,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
    FEW_SHOT_CNN_MODEL_NAME: ModelDefinition(
        name=FEW_SHOT_CNN_MODEL_NAME,
        backend="cnn",
        complexity_rank=9,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_few_shot_cnn,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
}


def create(
    name: str,
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None = None,
) -> Any:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    definition = _REGISTRY[name]
    if definition.backend == "sklearn":
        return _build_pipeline(definition, params, random_state, task_spec)
    return definition.estimator_factory(params, random_state, task_spec)


def candidate_params(
    name: str,
    cnn_hyperparams: Path | str | dict[str, Any] | None = None,
    autoencoder_hyperparams: Path | str | dict[str, Any] | None = None,
    few_shot_hyperparams: Path | str | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    if name in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME}:
        axes, _metadata = resolve_cnn_hyperparams(cnn_hyperparams)
        return [dict(params) for params in _grid(**axes)]
    if name == AUTOENCODER_KNN_MODEL_NAME:
        axes, _metadata = resolve_autoencoder_hyperparams(autoencoder_hyperparams)
        return [dict(params) for params in _grid(**axes)]
    if name == FEW_SHOT_CNN_MODEL_NAME:
        axes, _metadata = resolve_few_shot_hyperparams(few_shot_hyperparams)
        return [dict(params) for params in _grid(**axes)]
    return [dict(params) for params in _REGISTRY[name].param_grid]


def model_complexity_rank(name: str) -> int:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return int(_REGISTRY[name].complexity_rank)


def normalization_policy(name: str) -> str:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return str(_REGISTRY[name].normalization_policy)


def model_backend(name: str) -> str:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return str(_REGISTRY[name].backend)


def supported_representation_kinds(name: str) -> tuple[str, ...]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return tuple(str(x) for x in _REGISTRY[name].supported_representation_kinds)


def registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())
