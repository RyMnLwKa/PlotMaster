from typing import Dict, Any
from services.registry import Registry
from models.visualization_task import ChartTask


class APIAdapter:
    """Преобразует семантическое описание графика в описание конкретной библиотеки, используя YAML Registry."""

    def __init__(self, registry: Registry = None):
        self.registry = registry or Registry()

    def run(self, chart_task: ChartTask) -> Dict[str, Any]:
        spec = self.registry.get(chart_task.chart_type)
        backend = spec["backend"]
        function = spec["function"]
        mapping = spec.get("mapping", {})
        defaults = spec.get("defaults", {})
        pipeline = spec.get("pipeline", ["figure", "plot", "tight_layout", "save", "close"])

        kwargs = dict(defaults)
        semantic_keys = set()
        for semantic_key, api_key in mapping.items():
            if semantic_key in chart_task.semantic and chart_task.semantic[semantic_key] is not None:
                kwargs[api_key] = chart_task.semantic[semantic_key]
                semantic_keys.add(api_key)

        # семантические ключи без явного маппинга передаются как есть, если совпадают с именем kwargs
        for k, v in chart_task.semantic.items():
            if k not in mapping and k not in ("show_regression",):
                kwargs.setdefault(k, v)
                semantic_keys.add(k)

        return {
            "backend": backend,
            "function": function,
            "kwargs": kwargs,
            "pipeline": pipeline,
            "spec": spec,
            # kwargs, взятые из data-семантики (x/y/hue и т.п.) — их нельзя перезаписывать
            # общим "style" в ParameterResolver. Всё остальное в kwargs — registry-defaults
            # (bins/kde/alpha и т.п.), и ИМЕННО их обязан перезаписывать пользовательский style.
            "semantic_keys": semantic_keys,
        }
