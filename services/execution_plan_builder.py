import os
from typing import Dict, Any
from models.execution_plan import ChartExecutionPlan
from models.visualization_task import ChartTask
from models.plot_description import PlotDescription


class ExecutionPlanBuilder:
    """Строит ChartExecutionPlan из результатов Validator -> APIAdapter -> ParameterResolver ->
    ThemeManager -> PlotDescriptionResolver."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, chart_task: ChartTask, adapted: Dict[str, Any], theme: Dict[str, Any],
            plot_description: PlotDescription = None) -> ChartExecutionPlan:
        output_path = os.path.join(self.output_dir, f"{chart_task.id}.png")
        return ChartExecutionPlan(
            chart_id=chart_task.id,
            backend=adapted["backend"],
            function=adapted["function"],
            kwargs=adapted["kwargs"],
            theme=theme,
            steps=adapted["pipeline"],
            semantic=chart_task.semantic,
            output_path=output_path,
            source=(chart_task.metadata or {}).get("source"),
            model_source=(chart_task.metadata or {}).get("model_source"),
            plot_description=plot_description or PlotDescription(),
        )