import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any
from models.execution_plan import ChartExecutionPlan
from models.visualization_artifact import VisualizationArtifact


def render(plan: ChartExecutionPlan, df: pd.DataFrame, ax) -> Dict[str, Any]:
    """Рисует decision_regions на уже существующем ax. Бросает исключения наружу."""
    from mlxtend.plotting import plot_decision_regions
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder

    statistics = {}
    x_col = plan.semantic.get("x")
    y_col = plan.semantic.get("y")
    target_col = plan.semantic.get("target") or plan.semantic.get("color_by")

    if not (x_col and y_col and target_col):
        raise ValueError("decision_regions требует semantic.x, semantic.y и semantic.target")

    X = df[[x_col, y_col]].to_numpy(dtype=float)
    le = LabelEncoder()
    y_labels = le.fit_transform(df[target_col])

    clf = SVC(kernel=plan.kwargs.get("kernel", "rbf"))
    clf.fit(X, y_labels)

    plot_decision_regions(X, y_labels, clf=clf, ax=ax)
    pd_desc = plan.plot_description
    ax.set_xlabel(pd_desc.xlabel or x_col)
    ax.set_ylabel(pd_desc.ylabel or y_col)
    title = pd_desc.title or plan.semantic.get("title") or f"Decision regions: {x_col} vs {y_col}"
    ax.set_title(title, fontsize=pd_desc.font_size)

    statistics["accuracy"] = float(clf.score(X, y_labels))
    statistics["support_vectors"] = int(clf.support_vectors_.shape[0])
    statistics["samples"] = int(len(df))
    return statistics


def run(plan: ChartExecutionPlan, df: pd.DataFrame) -> VisualizationArtifact:
    """
    Поддерживает построение decision_regions с использованием mlxtend.plotting.plot_decision_regions.
    Ожидает semantic: x, y (два числовых признака) и target (категориальная/числовая метка класса).
    Обучает простую модель (по умолчанию SVC) на лету, если явно не указано иное — это ограничение
    демонстрационного бэкенда: для реальных сценариев модель должна прийти извне.
    """
    start = time.time()
    warnings = []
    statistics = {}
    status = "ok"

    try:
        plt.rcParams["figure.dpi"] = plan.theme.get("dpi", 120)
        fig_size = tuple(plan.theme.get("figure_size", [8, 5]))
        fig, ax = plt.subplots(figsize=fig_size)

        statistics = render(plan, df, ax)

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
