from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

# Как результирующие графики попадают к пользователю:
# - "separate"  — каждый ChartTask сохраняется своим отдельным PNG-файлом (как сейчас).
# - "dashboard" — все ChartTask'и рисуются как N подграфиков (axes) на одной общей
#   Figure/дашборде (сетка layout.columns x ceil(n/columns)) и сохраняются одним файлом.
OUTPUT_MODES = ("separate", "dashboard")


@dataclass
class ChartTask:
    id: str
    chart_type: str
    semantic: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан ChartTask:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualizationTask:
    layout: Dict[str, Any] = field(default_factory=dict)
    charts: List[ChartTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Новое поле: явно определяет, нужно ли сохранить N отдельных графиков
    # или нанести их на общий дашборд из N axes. По умолчанию сохраняем
    # прежнее поведение ("separate"), чтобы не ломать существующие пайплайны.
    output_mode: str = "separate"

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан VisualizationTask:\n {pformat(self.to_dict())}")

        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"output_mode='{self.output_mode}' недопустим, ожидается одно из {OUTPUT_MODES}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VisualizationTask":
        charts = [ChartTask(**c) for c in data.get("charts", [])]
        return VisualizationTask(
            layout=data.get("layout", {}),
            charts=charts,
            metadata=data.get("metadata", {}),
            output_mode=data.get("output_mode", "separate"),
        )
