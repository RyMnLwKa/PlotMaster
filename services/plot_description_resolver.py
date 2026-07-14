from typing import Any, Dict, Optional
from models.visualization_task import ChartTask
from models.plot_description import PlotDescription, PLOT_DESCRIPTION_KEYS


class PlotDescriptionResolver:
    """Собирает PlotDescription для одного ChartTask.

    Источники (по возрастанию приоритета):
    1. semantic.title  — fallback, если Planner по старой памяти положил title в semantic;
    2. chart_task.style — основной источник, сюда Planner обязан класть title/xlabel/ylabel/
       color/palette/legend/grid/font_size/annotations согласно обновлённому prompt'у;
    3. overrides — параметры, явно переданные пользователем в API/UI поверх того, что решил
       Planner (например, пользователь после генерации попросил "сделай точки красными" —
       это можно передать сюда, не перегенерируя весь VisualizationTask).
    """

    def __init__(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None):
        # overrides: chart_id -> dict с полями PlotDescription
        self.overrides = overrides or {}

    def run(self, chart_task: ChartTask) -> PlotDescription:
        style = chart_task.style or {}
        semantic = chart_task.semantic or {}
        metadata = chart_task.metadata or {}

        base_data = {k: style[k] for k in PLOT_DESCRIPTION_KEYS if k in style}
        if "title" not in base_data and semantic.get("title"):
            base_data["title"] = semantic["title"]
        # Planner иногда кладёт color_mapping в metadata вместо style (по смыслу это
        # "пояснение к графику", а не строгий контракт) — подстраховываемся fallback'ом,
        # чтобы явное пожелание пользователя по цветам не терялось молча.
        if "color_mapping" not in base_data and metadata.get("color_mapping"):
            base_data["color_mapping"] = metadata["color_mapping"]

        description = PlotDescription.from_dict(base_data)

        chart_override = self.overrides.get(chart_task.id)
        if chart_override:
            description = description.merge(PlotDescription.from_dict(chart_override))

        return description
