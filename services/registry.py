import os
import yaml
from typing import Dict, Any

REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry")


class Registry:
    """Загружает и кэширует YAML-спецификации графиков из папки registry/."""

    def __init__(self, registry_dir: str = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self._cache: Dict[str, Any] = {}

    def get(self, chart_type: str) -> Dict[str, Any]:
        if chart_type in self._cache:
            return self._cache[chart_type]
        path = os.path.join(self.registry_dir, f"{chart_type}.yaml")
        if not os.path.exists(path):
            raise ValueError(f"Тип графика '{chart_type}' не найден в registry ({path})")
        with open(path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        self._cache[chart_type] = spec
        return spec

    def available_types(self):
        return sorted(
            f[:-5] for f in os.listdir(self.registry_dir) if f.endswith(".yaml")
        )
