import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

# функции seaborn, сами создающие Figure (не принимают ax) — несовместимы с дашбордом
NO_AX_FUNCTIONS = {"pairplot", "jointplot"}


def _apply_theme(theme: Dict[str, Any]):
    sns.set_theme(style=theme.get("style", "whitegrid"), palette=theme.get("palette", "deep"))
    plt.rcParams["figure.dpi"] = theme.get("dpi", 120)
    plt.rcParams["font.family"] = theme.get("font", "DejaVu Sans")


def render(plan: ChartExecutionPlan, df: pd.DataFrame, ax, model=None, source_dict=None) -> Dict[str, Any]:
    """Рисует график plan на уже существующем ax (без создания/сохранения Figure).

    Используется и для одиночного файла (run()), и для дашборда (DashboardRenderer).
    Функции из NO_AX_FUNCTIONS (pairplot/jointplot) сюда не подходят — они сами создают
    Figure; для них нужно использовать run() напрямую (и не в режиме dashboard).
    Бросает исключения наружу — вызывающий код сам решает, как их обрабатывать.
    """
    if plan.function in NO_AX_FUNCTIONS:
        raise ValueError(
            f"'{plan.function}' сам создаёт Figure и не может быть отрисован на "
            f"существующем ax (несовместимо с дашбордом)"
        )

    statistics = {}
    func = getattr(sns, plan.function)
    kwargs = dict(plan.kwargs)
    pd_desc = plan.plot_description

    if plan.function == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        kwargs["data"] = numeric_df.corr()
        if pd_desc.palette:
            kwargs["cmap"] = pd_desc.palette
    else:
        kwargs["data"] = df
    kwargs["ax"] = ax

    # Явный цвет из PlotDescription имеет приоритет: если пользователь попросил
    # конкретный цвет, он важнее автоматической группировки по hue.
    if pd_desc.color:
        kwargs.pop("hue", None)
        kwargs["color"] = pd_desc.color
    elif pd_desc.color_mapping and "hue" in kwargs:
        # seaborn принимает palette как dict {категория: цвет} — это как раз то,
        # что нужно для "покрась setosa в розовый, versicolor в жёлтый..."
        kwargs["palette"] = pd_desc.color_mapping

    func(**kwargs)

    if plan.semantic.get("show_regression") and plan.function == "scatterplot":
        x = plan.semantic.get("x")
        y = plan.semantic.get("y")
        if x and y:
            sns.regplot(data=df, x=x, y=y, ax=ax, scatter=False, color="red")

    # legend: приоритет у per-chart PlotDescription.legend, иначе - глобальная тема
    want_legend = pd_desc.legend if pd_desc.legend is not None else plan.theme.get("legend", True)
    if not want_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    # grid: приоритет у per-chart PlotDescription.grid, иначе - глобальная тема
    want_grid = pd_desc.grid if pd_desc.grid is not None else plan.theme.get("grid", True)
    ax.grid(want_grid, **({"alpha": 0.3} if want_grid else {}))

    title = pd_desc.title or plan.semantic.get("title") or plan.kwargs.get("title")
    if title:
        ax.set_title(title, fontsize=pd_desc.font_size)

    xlabel = pd_desc.xlabel or plan.semantic.get("x")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=pd_desc.font_size)
    ylabel = pd_desc.ylabel or plan.semantic.get("y")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=pd_desc.font_size)

    for annotation in pd_desc.annotations:
        if annotation.x is not None and annotation.y is not None:
            ax.annotate(annotation.text, xy=(annotation.x, annotation.y))
        else:
            ax_x, ax_y, ha, va = _POSITION_COORDS.get(
                annotation.position, _POSITION_COORDS["top_right"]
            )
            ax.text(ax_x, ax_y, annotation.text, transform=ax.transAxes,
                    ha=ha, va=va, fontsize=pd_desc.font_size or 9)

    # базовая статистика
    x = plan.semantic.get("x")
    y = plan.semantic.get("y")
    if x and x in df.columns and pd.api.types.is_numeric_dtype(df[x]):
        statistics[f"{x}_mean"] = float(df[x].mean())
    if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        statistics[f"{y}_mean"] = float(df[y].mean())
    if x and y and x in df.columns and y in df.columns and \
       pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        statistics["correlation"] = float(df[x].corr(df[y]))
    statistics["samples"] = int(len(df))
    return statistics


def run(plan: ChartExecutionPlan, df: pd.DataFrame, model=None, source_dict=None) -> VisualizationArtifact:
    start = time.time()
    warnings = []
    statistics = {}
    status = "ok"

    try:
        _apply_theme(plan.theme)
        fig_size = tuple(plan.theme.get("figure_size", [8, 5]))

        if plan.function in NO_AX_FUNCTIONS:
            kwargs = dict(plan.kwargs)
            kwargs["data"] = df
            func = getattr(sns, plan.function)
            grid = func(**kwargs)
            fig = grid.fig
            statistics["samples"] = int(len(df))
        else:
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