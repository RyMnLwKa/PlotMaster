import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from models.execution_plan import ChartExecutionPlan
from models.visualization_artifact import VisualizationArtifact


def render(plan: ChartExecutionPlan, df: pd.DataFrame, ax, model=None) -> Dict[str, Any]:
    """Рисует decision_regions на уже существующем ax. Бросает исключения наружу.

    ВАЖНО: этот backend НЕ обучает модель. Он только рисует decision_regions по УЖЕ
    ОБУЧЕННОЙ модели, полученной извне (см. run()). Обучение — задача этапа анализа
    (Analyst), а не визуализации: Analyst сам решает архитектуру/гиперпараметры модели
    (SVM, логрегрессия, KNN и т.п.), обучает её на своих данных/сплите и передаёт готовый
    объект через emit(id, model) с type: model. Tool резолвит chart_task.metadata
    ['model_source'] в этот объект и передаёт его сюда параметром model — визуализация
    работает с любой моделью, реализующей .predict(), не зная, как именно она обучена.
    """
    from mlxtend.plotting import plot_decision_regions
    from sklearn.preprocessing import LabelEncoder

    if model is None:
        raise ValueError(
            "decision_regions требует уже обученную модель: укажи в metadata графика "
            "\"model_source\": \"<id>\", ссылаясь на DataObject type='model', который "
            "Analyst передал через emit(id, model). Этот backend модель не обучает."
        )
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Объект, переданный через model_source, не является обученной моделью "
            f"(нет метода .predict): {type(model)!r}"
        )

    statistics = {}
    x_col = plan.semantic.get("x")
    y_col = plan.semantic.get("y")
    target_col = plan.semantic.get("target") or plan.semantic.get("color_by")

    if not (x_col and y_col and target_col):
        raise ValueError("decision_regions требует semantic.x, semantic.y и semantic.target")

    X = df[[x_col, y_col]].to_numpy(dtype=float)

    # Кодирование целевой переменной в целочисленные метки — это подготовка данных для
    # отрисовки/легенды (plot_decision_regions и статистика accuracy ожидают числовые
    # метки классов), а НЕ обучение модели. Модель уже обучена Analyst'ом заранее; если
    # target в датасете нечисловой, применяем тот же детерминированный (по отсортированным
    # уникальным значениям) LabelEncoder, который, как ожидается, использовал и Analyst
    # при обучении модели на этом же столбце — поэтому кодировка совпадёт.
    target_series = df[target_col]
    if pd.api.types.is_numeric_dtype(target_series):
        y_labels = target_series.to_numpy()
    else:
        y_labels = LabelEncoder().fit_transform(target_series)

    plot_decision_regions(X, y_labels, clf=model, ax=ax)
    pd_desc = plan.plot_description
    ax.set_xlabel(pd_desc.xlabel or x_col)
    ax.set_ylabel(pd_desc.ylabel or y_col)
    title = pd_desc.title or plan.semantic.get("title") or f"Decision regions: {x_col} vs {y_col}"
    ax.set_title(title, fontsize=pd_desc.font_size)

    try:
        statistics["accuracy"] = float(model.score(X, y_labels))
    except Exception:
        pass  # не все модели/сценарии поддерживают .score с этими X/y — не фатально для графика
    if hasattr(model, "support_vectors_"):
        statistics["support_vectors"] = int(model.support_vectors_.shape[0])
    statistics["samples"] = int(len(df))
    return statistics


def run(plan: ChartExecutionPlan, df: pd.DataFrame, model=None) -> VisualizationArtifact:
    """
    Поддерживает построение decision_regions с использованием mlxtend.plotting.plot_decision_regions
    по уже обученной модели (см. render()). Ожидает semantic: x, y (два числовых признака) и
    target (категориальная/числовая метка класса), а также готовую модель, переданную Tool'ом
    через параметр model (резолвится из chart_task.metadata['model_source']).
    """
    start = time.time()
    warnings = []
    statistics = {}
    status = "ok"

    try:
        plt.rcParams["figure.dpi"] = plan.theme.get("dpi", 120)
        fig_size = tuple(plan.theme.get("figure_size", [8, 5]))
        fig, ax = plt.subplots(figsize=fig_size)

        statistics = render(plan, df, ax, model=model)

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
