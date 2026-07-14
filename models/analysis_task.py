from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import logging
from pprint import pformat

logger = logging.getLogger(__name__)


@dataclass
class OutputSpec:
    """
    Декларация ОДНОГО результата (emit) шага анализа, полученная из YAML-контракта.
    Именно отсюда AnalysisExecutor берёт type/description/aligned_with_dataset/columns
    при сборке DataObject — сам python-код лишь передаёт value через emit(id, value).
    """
    id: str
    type: str
    description: str
    aligned_with_dataset: Optional[bool] = None
    columns: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан OutputSpec:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisRequest:
    """
    Один шаг анализа.

    - outputs: YAML-контракт — что именно должно быть получено (id, type,
      description, aligned_with_dataset, columns) и зачем.
    - code: python-код, свободно сгенерированный Analyst'ом, реализующий этот
      контракт. Использует df, pandas/numpy/sklearn/scipy/statsmodels и
      emit(id, value) — БЕЗ метаданных, они берутся из outputs по id.
    """
    id: str
    purpose: str
    outputs: List[OutputSpec] = field(default_factory=list)
    code: str = ""

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан AnalysisRequest:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisPlan:
    """
    Опциональный этап, выполняемый ДО VisualizationTask. Если анализ не
    требуется запросом пользователя, Analyst возвращает AnalysisPlan с
    пустым списком requests — пайплайн просто пропускает этот этап.
    """
    requests: List[AnalysisRequest] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан AnalysisPlan:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AnalysisPlan":
        requests = []
        for r in data.get("requests", []):
            outputs = [OutputSpec(**o) for o in r.get("outputs", [])]
            requests.append(AnalysisRequest(
                id=r["id"],
                purpose=r.get("purpose", ""),
                outputs=outputs,
                code=r.get("code", ""),
            ))
        return AnalysisPlan(requests=requests, metadata=data.get("metadata", {}))
