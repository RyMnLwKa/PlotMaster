import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Any
from models.execution_plan import ExecutionPlan, ChartExecutionPlan
from models.visualization_artifact import VisualizationArtifact
from tools import matplotlib_backend, seaborn_backend, mlxtend_backend

BACKENDS = {
    "matplotlib": matplotlib_backend,
    "seaborn": seaborn_backend,
    "mlxtend": mlxtend_backend,
}

# backend'ы, чьи функции сами создают Figure и не могут быть нарисованы на чужом ax
# (см. seaborn_backend.NO_AX_FUNCTIONS) — такие графики в режиме dashboard рендерятся
# отдельным файлом каждый, с предупреждением.
_DASHBOARD_INCOMPATIBLE = getattr(seaborn_backend, "NO_AX_FUNCTIONS", set())


class Tool:
    """Исполнитель. Не принимает решений — выполняет ExecutionPlan над DataFrame согласно backend.

    extra_frames — те же самостоятельные таблицы анализа, что передавались в Executor.run,
    нужны здесь, чтобы каждый ChartExecutionPlan.source был построен по своей таблице.
    models — DataObjectStore.models(): {id: обученная модель}, нужны здесь, чтобы backend'ы,
    которым требуется уже обученная модель (например mlxtend decision_regions), получили
    готовый объект через chart_plan.model_source — сами backend'ы модель НЕ обучают.
    """

    def run(self, plan: ExecutionPlan, df: pd.DataFrame,
            extra_frames: Optional[Dict[str, pd.DataFrame]] = None,
            models: Optional[Dict[str, Any]] = None) -> List[VisualizationArtifact]:
        extra_frames = extra_frames or {}
        models = models or {}
        if plan.output_mode == "dashboard":
            return self._run_dashboard(plan, df, extra_frames, models)
        return self._run_separate(plan.charts, df, extra_frames, models)

    def _resolve_model(self, chart_plan: ChartExecutionPlan, models: Dict[str, Any]) -> Optional[Any]:
        if not chart_plan.model_source:
            return None
        model = models.get(chart_plan.model_source)
        if model is None:
            raise ValueError(
                f"[{chart_plan.chart_id}] metadata.model_source='{chart_plan.model_source}' "
                f"не найден среди результатов анализа: {list(models.keys())}"
            )
        return model

    def _run_separate(self, chart_plans: List[ChartExecutionPlan], df: pd.DataFrame,
                       extra_frames: Dict[str, pd.DataFrame],
                       models: Dict[str, Any]) -> List[VisualizationArtifact]:
        artifacts = []
        for chart_plan in chart_plans:
            chart_df = extra_frames.get(chart_plan.source, df) if chart_plan.source else df
            backend_module = BACKENDS.get(chart_plan.backend)
            if backend_module is None:
                artifacts.append(VisualizationArtifact(
                    chart_id=chart_plan.chart_id,
                    semantic=chart_plan.semantic,
                    image_path="",
                    backend=chart_plan.backend,
                    function=chart_plan.function,
                    execution_time=0.0,
                    status="error",
                    warnings=[f"Неизвестный backend '{chart_plan.backend}'"],
                ))
                continue
            try:
                model = self._resolve_model(chart_plan, models)
            except ValueError as e:
                artifacts.append(VisualizationArtifact(
                    chart_id=chart_plan.chart_id,
                    semantic=chart_plan.semantic,
                    image_path="",
                    backend=chart_plan.backend,
                    function=chart_plan.function,
                    execution_time=0.0,
                    status="error",
                    warnings=[str(e)],
                ))
                continue
            artifacts.append(backend_module.run(chart_plan, chart_df, model=model))
        return artifacts

    def _run_dashboard(self, plan: ExecutionPlan, df: pd.DataFrame,
                        extra_frames: Dict[str, pd.DataFrame],
                        models: Dict[str, Any]) -> List[VisualizationArtifact]:
        """Рисует все совместимые графики как N подграфиков (axes) на одной общей Figure
        и сохраняет их одним файлом (output/dashboard.png). Графики, чей backend/function
        не умеет рисовать на "чужом" ax (pairplot/jointplot), выпадают из дашборда и
        рендерятся отдельным файлом каждый — с предупреждением в артефакте."""
        dashboard_charts = [
            cp for cp in plan.charts
            if cp.function not in _DASHBOARD_INCOMPATIBLE and cp.backend in BACKENDS
        ]
        dashboard_ids = {id(cp) for cp in dashboard_charts}
        fallback_charts = [cp for cp in plan.charts if id(cp) not in dashboard_ids]

        artifacts = self._run_separate(fallback_charts, df, extra_frames, models)
        for a in artifacts:
            a.warnings = list(a.warnings) + [
                "график несовместим с output_mode='dashboard' — сохранён отдельным файлом"
            ]

        if not dashboard_charts:
            return artifacts

        n = len(dashboard_charts)
        columns = max(1, int((plan.layout or {}).get("columns", 1)))
        rows = math.ceil(n / columns)

        # Размер и dpi общей фигуры берём из темы первого графика (Planner обычно задаёт
        # одинаковый dpi/figure_size на весь набор, если просит общий дашборд).
        base_theme = dashboard_charts[0].theme
        cell_w, cell_h = base_theme.get("figure_size", [8, 5])
        dpi = base_theme.get("dpi", 120)

        fig, axes = plt.subplots(rows, columns, figsize=(cell_w * columns, cell_h * rows))
        axes_flat = axes.flatten().tolist() if n > 1 else [axes]

        dashboard_path = (dashboard_charts[0].output_path.rsplit("/", 1)[0] + "/dashboard.png") \
            if "/" in dashboard_charts[0].output_path else "dashboard.png"

        for ax in axes_flat[n:]:
            ax.axis("off")  # пустые ячейки сетки, если n не кратно columns

        for chart_plan, ax in zip(dashboard_charts, axes_flat):
            start = time.time()
            chart_df = extra_frames.get(chart_plan.source, df) if chart_plan.source else df
            backend_module = BACKENDS[chart_plan.backend]
            try:
                model = self._resolve_model(chart_plan, models)
                statistics = backend_module.render(chart_plan, chart_df, ax, model=model)
                artifacts.append(VisualizationArtifact(
                    chart_id=chart_plan.chart_id,
                    semantic=chart_plan.semantic,
                    image_path=dashboard_path,
                    backend=chart_plan.backend,
                    function=chart_plan.function,
                    execution_time=time.time() - start,
                    status="ok",
                    warnings=[],
                    statistics=statistics,
                ))
            except Exception as e:
                ax.axis("off")
                artifacts.append(VisualizationArtifact(
                    chart_id=chart_plan.chart_id,
                    semantic=chart_plan.semantic,
                    image_path="",
                    backend=chart_plan.backend,
                    function=chart_plan.function,
                    execution_time=time.time() - start,
                    status="error",
                    warnings=[str(e)],
                ))

        plt.tight_layout()
        fig.savefig(dashboard_path, dpi=dpi)
        plt.close(fig)

        return artifacts
