import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from models.execution_plan import ChartExecutionPlan
from models.visualization_artifact import VisualizationArtifact

_POSITION_COORDS = {
    "top_right": (0.98, 0.98, "right", "top"),
    "top_left": (0.02, 0.98, "left", "top"),
    "bottom_right": (0.98, 0.02, "right", "bottom"),
    "bottom_left": (0.02, 0.02, "left", "bottom"),
}

# kwargs-ключи, значения которых registry/Planner передают как ИМЯ СТОЛБЦА (строку),
# но которые matplotlib ожидает как массив данных, а не как строку. Можн обновлять при
# расширении registry/
ARRAY_COLUMN_KWARGS = {"yerr", "xerr", "labels"}

# функции, для которых x/y не подписываются как xlabel/ylabel и не красятся через
# pd_desc.color (у pie нет декартовых осей, а color для одного сектора не имеет смысла —
# для множества цветов есть kwarg "colors", который пользователь может передать явно
# через style/defaults).
_NO_AXES_LABELS_FUNCTIONS = {"pie"}

def _resolve_array_kwargs(kwargs: Dict[str, Any], df: pd.DataFrame) -> None:
    """Функция превращает имена колонок (переданные как строки)
    из kwargs, представленных в ARRAY_COLUMN_KWARGS в реальные данные из
    этих колонок (массивы/списки). Если kwarg уже не литерал, то не трогаем"""
    for key in ARRAY_COLUMN_KWARGS:
        if key not in kwargs:
            continue
        value = kwargs[key]
        if isinstance(value, str) and value in df.columns:
            kwargs[key] = df[value].tolist() if key == "labels" else df[value].to_numpy()



def render(plan: ChartExecutionPlan, df: pd.DataFrame, ax, model=None, source_dict=None) -> Dict[str, Any]:
    """Рисует график plan на уже существующем ax (без создания/сохранения Figure).

    Используется и для одиночного файла (run() создаёт свой ax и вызывает render),
    и для дашборда (DashboardRenderer передаёт сюда один из subplot'ов общей Figure).
    Бросает исключения наружу — вызывающий код сам решает, как их обрабатывать.
    Возвращает статистику по данным (samples, means и т.п.).
    """
    statistics = {}
    kwargs = dict(plan.kwargs)
    x = kwargs.pop("x", None)
    y = kwargs.pop("y", None)
    pd_desc = plan.plot_description

    _resolve_array_kwargs(kwargs, df)

    if pd_desc.color and plan.function not in _NO_AXES_LABELS_FUNCTIONS:
        kwargs["color"] = pd_desc.color

    func = getattr(ax, plan.function)
    args = []
    if x is not None and y is not None:
        args = [df[x], df[y]]
    elif x is not None:
        args = [df[x]]

    func(*args, **kwargs)

    title = pd_desc.title or plan.semantic.get("title")
    if title:
        ax.set_title(title, fontsize=pd_desc.font_size)

    if plan.function not in _NO_AXES_LABELS_FUNCTIONS:
        xlabel = pd_desc.xlabel or x
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=pd_desc.font_size)
        ylabel = pd_desc.ylabel or y
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=pd_desc.font_size)

    want_legend = pd_desc.legend if pd_desc.legend is not None else plan.theme.get("legend", True)
    if want_legend and ax.get_legend_handles_labels()[0]:
        ax.legend()

    want_grid = pd_desc.grid if pd_desc.grid is not None else plan.theme.get("grid", True)
    ax.grid(want_grid, **({"alpha": 0.3} if want_grid else {}))

    for annotation in pd_desc.annotations:
        if annotation.x is not None and annotation.y is not None:
            ax.annotate(annotation.text, xy=(annotation.x, annotation.y))
        else:
            ax_x, ax_y, ha, va = _POSITION_COORDS.get(
                annotation.position, _POSITION_COORDS["top_right"]
            )
            ax.text(ax_x, ax_y, annotation.text, transform=ax.transAxes,
                    ha=ha, va=va, fontsize=pd_desc.font_size or 9)

    if x and x in df.columns and pd.api.types.is_numeric_dtype(df[x]):
        statistics[f"{x}_mean"] = float(df[x].mean())
    statistics["samples"] = int(len(df))
    return statistics


def run(plan: ChartExecutionPlan, df: pd.DataFrame, model=None, source_dict=None) -> VisualizationArtifact:
    start = time.time()
    warnings = []
    statistics = {}
    status = "ok"

    try:
        plt.rcParams["figure.dpi"] = plan.theme.get("dpi", 120)
        plt.rcParams["font.family"] = plan.theme.get("font", "DejaVu Sans")
        fig_size = tuple(plan.theme.get("figure_size", [8, 5]))
        fig, ax = plt.subplots(figsize=fig_size)

        statistics = render(plan, df, ax, model=model, source_dict=source_dict)

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