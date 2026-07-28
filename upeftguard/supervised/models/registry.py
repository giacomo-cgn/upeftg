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

from ..contracts import (
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
TRANSFORMER_MODEL_NAME = "transformer"
TORCH_SEQUENCE_BACKEND = "torch_sequence"
_CNN_1D_INTEGER_HYPERPARAMS = {"conv_channels", "num_conv_layers", "kernel_size"}
_CNN_1D_FLOAT_HYPERPARAMS = {"dropout", "learning_rate", "weight_decay"}
TRANSFORMER_HYPERPARAM_NAMES = (
    "d_model",
    "nhead",
    "slot_num_layers",
    "layer_num_layers",
    "dim_feedforward",
    "dropout",
    "pooling",
    "learning_rate",
    "weight_decay",
)
_TRANSFORMER_INTEGER_HYPERPARAMS = {
    "d_model",
    "nhead",
    "slot_num_layers",
    "layer_num_layers",
    "dim_feedforward",
}
_TRANSFORMER_FLOAT_HYPERPARAMS = {"dropout", "learning_rate", "weight_decay"}
_TRANSFORMER_STRING_HYPERPARAMS = {"pooling"}
_PRE_NORMALIZED_INPUT_PARAM = "_upeftguard_pre_normalized_input"


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
    return tuple({key: value for key, value in zip(keys, combo)} for combo in product(*values))


def _passthrough() -> str:
    return "passthrough"


def _standard_scaler() -> StandardScaler:
    return StandardScaler()


def _load_json_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Hyperparameter JSON not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Hyperparameter JSON must be an object at the top level, got {type(payload).__name__}")
    return payload


def _load_hyperparameter_payload(
    hyperparams: Path | str | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(hyperparams, dict):
        return dict(hyperparams), {"source": "inline_object", "path": None}
    source_path = Path(hyperparams).expanduser().resolve()
    return _load_json_object(source_path), {"source": "file", "path": str(source_path)}


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
        raise ValueError("CNN and DANN models require --hyperparams with an explicit hyperparameter JSON")
    payload, source = _load_hyperparameter_payload(cnn_hyperparams)

    axes = _normalize_cnn_hyperparam_axes(payload)
    metadata = {
        **source,
        "axes": {name: list(values) for name, values in axes.items()},
        "n_candidates": int(len(_grid(**axes))),
    }
    return axes, metadata


def _normalize_transformer_hyperparam_axes(payload: dict[str, Any]) -> dict[str, list[Any]]:
    raw_keys = {str(key) for key in payload.keys()}
    expected_keys = set(TRANSFORMER_HYPERPARAM_NAMES)
    optional_keys = {"pooling"}
    missing = sorted(expected_keys - optional_keys - raw_keys)
    extra = sorted(raw_keys - expected_keys)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "Transformer hyperparameter JSON must define the transformer axes; "
            "'pooling' is optional and defaults to ['mean']: "
            f"{list(TRANSFORMER_HYPERPARAM_NAMES)} ({'; '.join(details)})"
        )

    normalized: dict[str, list[Any]] = {}
    for name in TRANSFORMER_HYPERPARAM_NAMES:
        raw_values = payload.get(name, ["mean"] if name == "pooling" else None)
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                f"Transformer hyperparameter '{name}' must be a non-empty JSON list, got {type(raw_values).__name__}"
            )
        if name in _TRANSFORMER_INTEGER_HYPERPARAMS:
            normalized[name] = [int(value) for value in raw_values]
        elif name in _TRANSFORMER_FLOAT_HYPERPARAMS:
            normalized[name] = [float(value) for value in raw_values]
        elif name in _TRANSFORMER_STRING_HYPERPARAMS:
            pooling_values = [str(value) for value in raw_values]
            unsupported = sorted(set(pooling_values) - {"mean", "mean_max"})
            if unsupported:
                raise ValueError(f"Transformer pooling must be one of ['mean', 'mean_max'], got {unsupported}")
            normalized[name] = pooling_values
        else:
            raise ValueError(f"Unsupported transformer hyperparameter axis {name!r}")
    return normalized


def resolve_transformer_hyperparams(
    transformer_hyperparams: Path | str | dict[str, Any] | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if transformer_hyperparams is None:
        raise ValueError("Transformer models require --hyperparams with an explicit hyperparameter JSON")
    payload, source = _load_hyperparameter_payload(transformer_hyperparams)

    axes = _normalize_transformer_hyperparam_axes(payload)
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
    normalizer = (
        _passthrough() if bool(params.get(_PRE_NORMALIZED_INPUT_PARAM, False)) else definition.normalization_factory()
    )
    return Pipeline(
        steps=[
            ("normalizer", normalizer),
            ("model", definition.estimator_factory(params, random_state, task_spec)),
        ]
    )


def _create_cnn_1d(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .cnn.estimator import CNN1DSupervisedModel

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
        normalize_input_features=not bool(params.get(_PRE_NORMALIZED_INPUT_PARAM, False)),
    )


def _create_cnn_1d_dann(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .cnn.dann import CNN1DDANNSupervisedModel

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
        class_weight_loss=bool(params.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(params.get("rank_label_weight_loss", False)),
        normalize_input_features=not bool(params.get(_PRE_NORMALIZED_INPUT_PARAM, False)),
    )


def _create_transformer(
    params: dict[str, Any],
    random_state: int,
    task_spec: SupervisedTaskSpec | None,
) -> Any:
    from .transformer import TransformerSupervisedModel

    return TransformerSupervisedModel(
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        slot_num_layers=int(params["slot_num_layers"]),
        layer_num_layers=int(params["layer_num_layers"]),
        dim_feedforward=int(params["dim_feedforward"]),
        dropout=float(params["dropout"]),
        norm_first=True,
        pooling=str(params.get("pooling", "mean")),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        random_state=int(random_state),
        task_spec=task_spec,
        class_weight_loss=bool(params.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(params.get("rank_label_weight_loss", False)),
        normalize_input_features=not bool(params.get(_PRE_NORMALIZED_INPUT_PARAM, False)),
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
    "lasso": ModelDefinition(
        name="lasso",
        backend="sklearn",
        complexity_rank=1,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state, task_spec: LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            solver="saga",
            l1_ratio=1.0,
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
        backend=TORCH_SEQUENCE_BACKEND,
        complexity_rank=6,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_cnn_1d,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
    CNN_1D_DANN_MODEL_NAME: ModelDefinition(
        name=CNN_1D_DANN_MODEL_NAME,
        backend=TORCH_SEQUENCE_BACKEND,
        complexity_rank=7,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_cnn_1d_dann,
        supported_representation_kinds=(ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,),
    ),
    TRANSFORMER_MODEL_NAME: ModelDefinition(
        name=TRANSFORMER_MODEL_NAME,
        backend=TORCH_SEQUENCE_BACKEND,
        complexity_rank=8,
        normalization_policy="masked_train_only",
        normalization_factory=_passthrough,
        param_grid=(),
        estimator_factory=_create_transformer,
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


def _registered_param_axes(name: str) -> dict[str, list[Any]]:
    grid = _REGISTRY[name].param_grid
    if not grid:
        return {}
    axes: dict[str, list[Any]] = {key: [] for key in grid[0]}
    for params in grid:
        if set(params) != set(axes):
            raise ValueError(f"Registered hyperparameter grid for {name!r} has inconsistent keys")
        for key, value in params.items():
            if value not in axes[key]:
                axes[key].append(value)
    return axes


def _normalize_classical_hyperparam_axes(name: str, payload: dict[str, Any]) -> dict[str, list[Any]]:
    expected_axes = _registered_param_axes(name)
    raw_keys = {str(key) for key in payload}
    expected_keys = set(expected_axes)
    missing = sorted(expected_keys - raw_keys)
    extra = sorted(raw_keys - expected_keys)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            f"Hyperparameter JSON for {name!r} must define exactly {sorted(expected_keys)} "
            f"({'; '.join(details)})"
        )
    axes: dict[str, list[Any]] = {}
    for key in expected_axes:
        values = payload.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Hyperparameter {key!r} for {name!r} must be a non-empty JSON list, "
                f"got {type(values).__name__}"
            )
        axes[key] = list(values)
    return axes


def resolve_model_hyperparams(
    name: str,
    hyperparams: Path | str | dict[str, Any] | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    if name in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME}:
        axes, metadata = resolve_cnn_hyperparams(hyperparams)
    elif name == TRANSFORMER_MODEL_NAME:
        axes, metadata = resolve_transformer_hyperparams(hyperparams)
    elif hyperparams is None:
        axes = _registered_param_axes(name)
        metadata = {"source": "registered_grid", "path": None}
    else:
        payload, metadata = _load_hyperparameter_payload(hyperparams)
        axes = _normalize_classical_hyperparam_axes(name, payload)
    return axes, {
        **metadata,
        "model_name": name,
        "axes": {key: list(values) for key, values in axes.items()},
        "n_candidates": int(len(_grid(**axes))),
    }


def candidate_params(
    name: str,
    hyperparams: Path | str | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    axes, _metadata = resolve_model_hyperparams(name, hyperparams)
    return [dict(params) for params in _grid(**axes)]


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


def validate_hyperparams(
    *,
    model_name: str,
    hyperparams: Path | str | dict[str, Any] | None,
) -> None:
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{model_name}'. Registered: {sorted(_REGISTRY.keys())}")
    if model_name in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME, TRANSFORMER_MODEL_NAME} and hyperparams is None:
        raise ValueError(f"--hyperparams is required when --model is {model_name}")
    if hyperparams is not None:
        resolve_model_hyperparams(model_name, hyperparams)
