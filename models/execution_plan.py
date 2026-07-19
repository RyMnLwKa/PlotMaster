from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from models.plot_description import PlotDescription

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class ChartExecutionPlan:
    chart_id: str
    backend: str
    function: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    theme: Dict[str, Any] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    semantic: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    # id самостоятельной DataObject-таблицы (см. DataObjectStore.standalone_frames),
    # если график строится не по основному DataFrame, а по результату анализа
    # (например, feature_importance или correlation_ranking). None = основной df.
    source: str = None
    # id DataObject'а типа "model" (см. DataObjectStore.models), если backend'у визуализации
    # нужна уже ОБУЧЕННАЯ Analyst'ом модель (например mlxtend decision_regions). Backend
    # получает готовый объект модели через Tool — сам он модель не обучает.
    model_source: str = None
    # Явное, типизированное описание оформления ЭТОГО графика (title/xlabel/ylabel/
    # color/annotations и т.д.). Backend обязан применить каждое непустое поле.
    plot_description: PlotDescription = field(default_factory=PlotDescription)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан ChartExecutionPlan:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    charts: List[ChartExecutionPlan] = field(default_factory=list)
    # Прокинуты из VisualizationTask, чтобы Tool знал, собирать ли общий дашборд.
    output_mode: str = "separate"
    layout: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан ExecutionPlan:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)