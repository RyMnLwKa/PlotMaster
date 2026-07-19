import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Tuple, List
from models.execution_plan import ChartExecutionPlan
from models.visualization_artifact import VisualizationArtifact
from models.function_spec import FunctionSpec
"""
Generic, registry-driven mlxtend backend.

mlxtend.plotting не имеет единой сигнатуры (в отличие от seaborn, где почти все функции
принимают data=/x=/y=/hue=/ax=) — plot_decision_regions, plot_confusion_matrix и
plot_learning_curves ждут совершенно разные аргументы. Поэтому унифицировать можно только
"конверт" (figure/theme/labels/save/statistics/error-handling), а не сам вызов функции.

Паттерн: для каждой поддерживаемой function из registry/*.yaml есть один FunctionSpec —
маленький адаптер, который знает, как из ChartExecutionPlan (kwargs/semantic из YAML mapping,
уже готовые df/model) собрать вызов конкретной mlxtend-функции. render()/run()
сами дальше НИЧЕГО не знают про конкретные функции — это просто диспетчер + общий конверт.

Backend НЕ обучает модели и не принимает ML-решений (train_test_split и т.п.) — все данные,
специфичные для функции, приходят готовыми через YAML mapping (semantic/kwargs) и через уже
существующий механизм model_source (см. models/execution_plan.py), который лишь резолвит уже
обученную Analyst'ом модель. Для learning_curves X_train/y_train/X_test/y_test могут быть:
именем столбца (или списком имён столбцов для X_*) в df, ИЛИ, если train/test разной длины
и их нельзя положить в один DataFrame, ссылкой на DataObject type='dict' (см. metadata.source и
DataObjectStore.dicts()) — тогда backend получает его как source_dict и берёт значения прямо
из него по тем же ключам X_train/y_train/X_test/y_test. Единственное вычисление, которое
backend всё же делает сам, — confusion matrix по двум СТОЛБЦАМ (y_true/y_pred), это чистая
метрика по данным, а не ML-решение.
"""


def _default_labels(plan: ChartExecutionPlan, ax) -> None:
    pd_desc = plan.plot_description
    x_col = plan.semantic.get("x")
    y_col = plan.semantic.get("y")
    if pd_desc.xlabel or x_col:
        ax.set_xlabel(pd_desc.xlabel or x_col)
    if pd_desc.ylabel or y_col:
        ax.set_ylabel(pd_desc.ylabel or y_col)
    title = pd_desc.title or plan.semantic.get("title")
    if title:
        ax.set_title(title, fontsize=pd_desc.font_size)


def _title_only(plan: ChartExecutionPlan, ax) -> None:
    pd_desc = plan.plot_description
    title = pd_desc.title or plan.semantic.get("title")
    if title:
        ax.set_title(title, fontsize=pd_desc.font_size)


def _build_decision_regions(plan, df, model, source_dict=None):
    from sklearn.preprocessing import LabelEncoder

    if model is None or not hasattr(model, "predict"):
        raise ValueError(
            "decision_regions требует уже обученную модель: укажи в metadata графика "
            "\"model_source\": \"<id>\", ссылаясь на DataObject type='model', который "
            "Analyst передал через emit(id, model). Этот backend модель не обучает."
        )
    x_col = plan.semantic.get("x")
    y_col = plan.semantic.get("y")
    target_col = plan.semantic.get("target") or plan.semantic.get("color_by")
    if not (x_col and y_col and target_col):
        raise ValueError("decision_regions требует semantic.x, semantic.y и semantic.target")

    X = df[[x_col, y_col]].to_numpy(dtype=float)
    target_series = df[target_col]
    if pd.api.types.is_numeric_dtype(target_series):
        y_labels = target_series.to_numpy()
    else:
        y_labels = LabelEncoder().fit_transform(target_series)

    # прокидываем в _stats через замыкание на plan.kwargs, чтобы не пересчитывать X/y дважды
    plan.kwargs["_resolved_X"] = X
    plan.kwargs["_resolved_y"] = y_labels
    return (X, y_labels), {"clf": model}


def _stats_decision_regions(plan, df, model, result) -> Dict[str, Any]:
    statistics = {"samples": int(len(df))}
    X = plan.kwargs.pop("_resolved_X", None)
    y = plan.kwargs.pop("_resolved_y", None)
    try:
        if X is not None and y is not None:
            statistics["accuracy"] = float(model.score(X, y))
    except Exception:
        pass  # не все модели/сценарии поддерживают .score с этими X/y — не фатально
    if hasattr(model, "support_vectors_"):
        statistics["support_vectors"] = int(model.support_vectors_.shape[0])
    return statistics


def _build_confusion_matrix(plan, df, model, source_dict=None):
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    y_true_col = plan.kwargs.get("y_true") or plan.semantic.get("y_true")
    y_pred_col = plan.kwargs.get("y_pred") or plan.semantic.get("y_pred")
    if not (y_true_col and y_pred_col):
        raise ValueError(
            "plot_confusion_matrix требует semantic.y_true и semantic.y_pred — имена столбцов "
            "с фактическими и предсказанными метками (backend сам считает confusion matrix "
            "по этим двум столбцам, ML-решений не принимает)."
        )
    for col in (y_true_col, y_pred_col):
        if col not in df.columns:
            raise ValueError(f"plot_confusion_matrix: столбец '{col}' отсутствует в датасете")

    class_names = plan.kwargs.get("class_names") or plan.semantic.get("class_names")
    cm = sk_confusion_matrix(df[y_true_col], df[y_pred_col])

    kwargs = {k: v for k, v in plan.kwargs.items()
              if k not in ("y_true", "y_pred", "class_names")}
    if class_names:
        kwargs["class_names"] = class_names
    kwargs["conf_mat"] = cm
    plan.kwargs["_resolved_cm"] = cm
    return (), kwargs


def _stats_confusion_matrix(plan, df, model, result) -> Dict[str, Any]:
    cm = plan.kwargs.pop("_resolved_cm", None)
    statistics = {"samples": int(len(df))}
    if cm is not None:
        total = cm.sum()
        correct = np.trace(cm)
        statistics["accuracy"] = float(correct / total) if total else None
        statistics["classes"] = int(cm.shape[0])
    return statistics


def _build_learning_curves(plan, df, model, source_dict=None):
    if model is None or not hasattr(model, "predict"):
        raise ValueError(
            "plot_learning_curves требует уже обученную модель: укажи в metadata графика "
            "\"model_source\": \"<id>\". Этот backend модель не обучает и не делает "
            "train_test_split — он лишь рисует уже готовый train/test."
        )

    X_train_raw = plan.kwargs.get("X_train") or plan.semantic.get("X_train")
    y_train_raw = plan.kwargs.get("y_train") or plan.semantic.get("y_train")
    X_test_raw = plan.kwargs.get("X_test") or plan.semantic.get("X_test")
    y_test_raw = plan.kwargs.get("y_test") or plan.semantic.get("y_test")

    # Если metadata.source ссылается на DataObject type='dict' (train/test разной длины,
    # их нельзя было положить в один DataFrame) — Tool уже резолвил его в source_dict.
    # Тогда X_train/y_train/X_test/y_test в semantic — это просто ключи ("X_train" и т.п.),
    # а сами значения нужно брать из source_dict, а не искать как столбцы df.
    if source_dict is not None:
        missing = [k for k in ("X_train", "y_train", "X_test", "y_test") if k not in source_dict]
        if missing:
            raise ValueError(
                f"plot_learning_curves: в metadata.source словаре (type='dict') отсутствуют "
                f"ключи {missing} — ожидаются X_train/y_train/X_test/y_test."
            )
        X_train_raw, y_train_raw = source_dict["X_train"], source_dict["y_train"]
        X_test_raw, y_test_raw = source_dict["X_test"], source_dict["y_test"]

    if not (X_train_raw is not None and y_train_raw is not None
            and X_test_raw is not None and y_test_raw is not None):
        raise ValueError(
            "plot_learning_curves требует semantic.X_train/y_train/X_test/y_test — каждое "
            "значение может быть именем столбца (или списком имён столбцов для X_*) в df, "
            "уже готовым списком/массивом значений, либо (если train/test разной длины) "
            "ссылкой через metadata.source на DataObject type='dict' с этими же ключами."
        )

    def _resolve(value, label, is_X):
        """value может быть: (1) именем столбца в df, (2) списком имён столбцов
        (только для X_*), (3) уже готовым list/ndarray со значениями (в т.ч. из source_dict)."""
        if isinstance(value, str):
            if value not in df.columns:
                raise ValueError(f"plot_learning_curves: столбец '{label}'='{value}' отсутствует в датасете")
            arr = df[[value]].to_numpy(dtype=float) if is_X else df[value].to_numpy()
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            missing = [c for c in value if c not in df.columns]
            if missing:
                raise ValueError(f"plot_learning_curves: столбцы {label}={missing} отсутствуют в датасете")
            arr = df[list(value)].to_numpy(dtype=float) if is_X else df[list(value)].to_numpy()
        else:
            # уже готовые данные (list/tuple/ndarray с значениями, не имена столбцов)
            arr = np.asarray(value, dtype=float if is_X else None)
            if is_X and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        return arr

    X_train = _resolve(X_train_raw, "X_train", is_X=True)
    y_train = _resolve(y_train_raw, "y_train", is_X=False)
    X_test = _resolve(X_test_raw, "X_test", is_X=True)
    y_test = _resolve(y_test_raw, "y_test", is_X=False)

    if len(X_train) != len(y_train):
        raise ValueError(f"plot_learning_curves: X_train и y_train разной длины ({len(X_train)} vs {len(y_train)})")
    if len(X_test) != len(y_test):
        raise ValueError(f"plot_learning_curves: X_test и y_test разной длины ({len(X_test)} vs {len(y_test)})")

    kwargs = {k: v for k, v in plan.kwargs.items()
              if k not in ("X_train", "y_train", "X_test", "y_test")}
    kwargs["clf"] = model
    kwargs.setdefault("suppress_plot", False)
    kwargs.setdefault("print_model", False)
    return (X_train, y_train, X_test, y_test), kwargs


def _stats_learning_curves(plan, df, model, result) -> Dict[str, Any]:
    # plot_learning_curves возвращает (train_errors, test_errors)
    statistics = {}
    if isinstance(result, tuple) and len(result) == 2:
        train_errors, test_errors = result
        if train_errors:
            statistics["final_train_error"] = float(train_errors[-1])
        if test_errors:
            statistics["final_test_error"] = float(test_errors[-1])
    return statistics


FUNCTION_HANDLERS: Dict[str, FunctionSpec] = {
    "plot_decision_regions": FunctionSpec(
        requires_model=True,
        ax_compatible=True,
        build_call=_build_decision_regions,
        apply_labels=_default_labels,
        compute_statistics=_stats_decision_regions,
    ),
    "plot_confusion_matrix": FunctionSpec(
        requires_model=False,
        ax_compatible=True,
        build_call=_build_confusion_matrix,
        apply_labels=_title_only,
        compute_statistics=_stats_confusion_matrix,
    ),
    "plot_learning_curves": FunctionSpec(
        requires_model=True,
        ax_compatible=False,
        build_call=_build_learning_curves,
        apply_labels=_title_only,
        compute_statistics=_stats_learning_curves,
    ),
}

# функции, сами создающие Figure (без ax) — несовместимы с dashboard, см. tools/tool.py
NO_AX_FUNCTIONS = {name for name, spec in FUNCTION_HANDLERS.items() if not spec.ax_compatible}


def _get_spec(plan: ChartExecutionPlan) -> FunctionSpec:
    spec = FUNCTION_HANDLERS.get(plan.function)
    if spec is None:
        raise ValueError(
            f"mlxtend-функция '{plan.function}' не поддержана backend'ом. Добавь для неё "
            f"FunctionSpec в tools/mlxtend_backend.py::FUNCTION_HANDLERS."
        )
    return spec


def render(plan: ChartExecutionPlan, df: pd.DataFrame, ax, model=None, source_dict=None) -> Dict[str, Any]:
    """Рисует любую зарегистрированную mlxtend-функцию на уже существующем ax (см. FUNCTION_HANDLERS).
    Бросает исключения наружу — вызывающий код сам решает, как их обрабатывать."""
    import mlxtend.plotting as mlx

    spec = _get_spec(plan)
    if not spec.ax_compatible:
        raise ValueError(
            f"'{plan.function}' сам создаёт Figure и не может быть отрисован на существующем "
            f"ax (несовместимо с дашбордом) — используй run() напрямую, не в режиме dashboard"
        )
    if spec.requires_model and (model is None or not hasattr(model, "predict")):
        raise ValueError(
            f"'{plan.function}' требует уже обученную модель через metadata.model_source "
            f"(DataObject type='model'); backend модель не обучает."
        )

    func = getattr(mlx, plan.function)
    args, kwargs = spec.build_call(plan, df, model, source_dict)
    kwargs["ax"] = ax
    result = func(*args, **kwargs)

    if spec.apply_labels:
        spec.apply_labels(plan, ax)

    statistics = {"samples": int(len(df))}
    if spec.compute_statistics:
        statistics.update(spec.compute_statistics(plan, df, model, result))
    return statistics


def run(plan: ChartExecutionPlan, df: pd.DataFrame, model=None, source_dict=None) -> VisualizationArtifact:
    """Строит один PNG для любой зарегистрированной mlxtend-функции (см. FUNCTION_HANDLERS)."""
    import mlxtend.plotting as mlx

    start = time.time()
    warnings: List[str] = []
    statistics: Dict[str, Any] = {}
    status = "ok"

    try:
        spec = _get_spec(plan)
        plt.rcParams["figure.dpi"] = plan.theme.get("dpi", 120)
        fig_size = tuple(plan.theme.get("figure_size", [8, 5]))

        if spec.requires_model and (model is None or not hasattr(model, "predict")):
            raise ValueError(
                f"'{plan.function}' требует уже обученную модель через metadata.model_source "
                f"(DataObject type='model'); backend модель не обучает."
            )

        if spec.ax_compatible:
            fig, ax = plt.subplots(figsize=fig_size)
            statistics = render(plan, df, ax, model=model, source_dict=source_dict)
        else:
            # функция сама создаёт свою Figure (например plot_learning_curves) — так же,
            # как seaborn_backend уже делает для pairplot/jointplot (NO_AX_FUNCTIONS)
            func = getattr(mlx, plan.function)
            args, kwargs = spec.build_call(plan, df, model, source_dict)
            result = func(*args, **kwargs)
            fig = plt.gcf()
            ax = plt.gca()
            if spec.apply_labels:
                spec.apply_labels(plan, ax)
            statistics = {"samples": int(len(df))}
            if spec.compute_statistics:
                statistics.update(spec.compute_statistics(plan, df, model, result))

        plt.tight_layout()
        fig.savefig(plan.output_path, dpi=plan.theme.get("dpi", 120))
        plt.close(fig)

    except Exception as e:
        status = "error"
        warnings.append(str(e))

    return VisualizationArtifact(
        chart_id=plan.chart_id,
        semantic=plan.semantic,
        image_path=plan.output_path if status == "ok" else "",
        backend=plan.backend,
        function=plan.function,
        execution_time=time.time() - start,
        status=status,
        warnings=warnings,
        statistics=statistics,
    )