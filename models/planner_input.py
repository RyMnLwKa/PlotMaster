from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
from models.dataset_context import DatasetContext

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class PlannerInput:
    prompt: str
    dataset_context: DatasetContext
    # Краткое описание DataObject'ов, полученных на этапе AnalysisPlan (если он выполнялся).
    # Каждый элемент: {"id", "type", "metadata", "usable_as_column": bool}.
    # Planner может ссылаться на такие id как на обычные столбцы в semantic (x, y, color_by и т.д.),
    # т.к. DataObjectStore материализует их в рабочий DataFrame перед Executor'ом.
    analysis_context: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан PlannerInput:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "dataset_context": self.dataset_context.to_dict(),
            "analysis_context": self.analysis_context,
        }
