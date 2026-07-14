from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class DatasetContext:
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    null_counts: Dict[str, int]
    null_percent: Dict[str, float]
    unique_counts: Dict[str, int]
    sample: List[Dict[str, Any]]
    column_info: Dict[str, Dict[str, Any]]
    target: Optional[str] = None

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан DatasetContext:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
