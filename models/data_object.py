from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import logging
from functools import wraps
from pprint import pformat

logger = logging.getLogger(__name__)

ALLOWED_TYPES = ("series", "dataframe", "metrics", "model", "array", "dict")

def log_func(level=logging.DEBUG):  # Применяется к любой функции класса (преимущественно __init__ или to_dict)
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):

            obj_name = self.__class__.__name__
            result = func(self, *args, **kwargs)
            if logger.isEnabledFor(level):
                logger.debug(f" Создан {obj_name}:\n {pformat(self.to_dict())}")

            return result
        return wrapper
    return decorator

def log_cls(level=logging.DEBUG): # Применяется напрямую к классу, изменяя __init__ (подмеенивая его на new_init) и возвращая обновлённый класс
    def decorator(cls):
        original_init = cls.__init__
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            obj_name = self.__class__.__name__
            original_init(self, *args, **kwargs)
            if logger.isEnabledFor(level):
                logger.debug(f" Создан {obj_name}:\n {pformat(self.to_dict())}")
        cls.__init__ = new_init
        return cls
    return decorator

@log_cls(logging.DEBUG)
@dataclass
class DataObject:
    id: str
    type: str
    value: Any
    description: str
    aligned_with_dataset: Optional[bool] = None
    columns: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_step_id: str = ""
    status: str = "ok"
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> Optional[str]:
        if self.type not in ALLOWED_TYPES:
            return f"Недопустимый type '{self.type}'. Разрешены: {ALLOWED_TYPES}"

        if not self.description or not str(self.description).strip():
            return "description не должен быть пустой строкой"

        if self.type in ("series", "dataframe") and self.aligned_with_dataset is None:
            return f"Для type='{self.type}' обязателен aligned_with_dataset (True/False)"

        if self.type == "series" and not isinstance(self.value, pd.Series):
            return "Для type='series' value должен быть pd.Series"

        if self.type == "dataframe":
            if not isinstance(self.value, pd.DataFrame):
                return "Для type='dataframe' value должен быть pd.DataFrame"
            if self.columns is None:
                return "Для type='dataframe' обязателен columns"
            actual_columns = list(self.value.columns)
            if list(self.columns) != actual_columns:
                return (
                    f"columns {self.columns} не совпадает с фактическими "
                    f"столбцами value {actual_columns}"
                )

        if self.type == "dict":
            if not isinstance(self.value, dict):
                return "Для type='dict' value должен быть dict"

        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_data_object(obj: "DataObject") -> Optional[str]:
    return obj.validate()


def to_error_object(obj_id: str, description: str, error_message: str,
                     source_step_id: str = "") -> "DataObject":
    return DataObject(
        id=obj_id,
        type="metrics",
        value={},
        description=description,
        status="error",
        warnings=[error_message],
        source_step_id=source_step_id,
    )