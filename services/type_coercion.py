"""
Автоматическое приведение типов для параметров графиков (kwargs), приходящих из
YAML/JSON-конфигов и от Planner/Analyst-агентов.

Проблема: LLM-агент или YAML-конфиг иногда кладёт число как строку (bins: "30"),
булево значение как строку ("true"/"false"), либо просто нетипизированный текст.
Библиотечные функции (seaborn/matplotlib) при этом падают с ошибками вида
`[✗] chart_1: ['bins must be an integer, a string, or an array']`, потому что
получают "30" (str), а не 30 (int).

Поскольку заранее неизвестно, какие параметры примет конкретная функция бэкенда
(they are arbitrary kwargs, разные для каждого chart_type/registry-файла), мы не
можем построить строгую Pydantic-схему "поле -> ожидаемый тип" для каждого из них.
Вместо этого используем Pydantic для ПОСЛЕДОВАТЕЛЬНОЙ (каскадной) валидации/приведения
каждого строкового значения к наиболее вероятному типу, в порядке:

    bool -> int -> float -> str (str — гарантированный fallback, всегда проходит)

Значения, которые уже пришли типизированными (int/float/bool/None/list/dict/...),
не трогаем — приведение применяется только к строкам.
"""

from typing import Any, Dict

from pydantic import TypeAdapter, ValidationError

_BOOL_ADAPTER = TypeAdapter(bool)
_INT_ADAPTER = TypeAdapter(int)
_FLOAT_ADAPTER = TypeAdapter(float)
_STR_ADAPTER = TypeAdapter(str)

# Строковые представления булевых значений, которые распознаём явно (Pydantic сам
# по себе тоже принимает "true"/"false", но мы фиксируем список сами, чтобы не
# приводить произвольные слова вроде "on"/"off" ошибочно, если это часть данных).
_TRUE_STRINGS = {"true", "yes"}
_FALSE_STRINGS = {"false", "no"}


def coerce_scalar(value: Any) -> Any:
    """Приводит одно значение к наиболее вероятному типу.

    Приводим ТОЛЬКО строки (str). Всё остальное (int, float, bool, None, list,
    dict, объекты) возвращается как есть — например list уже является валидным
    типом для параметра bins ('bins must be an integer, a string, or an array').
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return value

    # 1) bool — только по явному словарю "истинных"/"ложных" строк.
    lowered = stripped.lower()
    if lowered in _TRUE_STRINGS or lowered in _FALSE_STRINGS:
        try:
            return _BOOL_ADAPTER.validate_python(lowered in _TRUE_STRINGS)
        except ValidationError:
            pass  # практически недостижимо, но не считаем это фатальной ошибкой

    # 2) int
    try:
        return _INT_ADAPTER.validate_python(stripped)
    except ValidationError:
        pass

    # 3) float
    try:
        return _FLOAT_ADAPTER.validate_python(stripped)
    except ValidationError:
        pass

    # 4) str — гарантированный fallback: возвращаем исходную строку без изменений.
    return _STR_ADAPTER.validate_python(value)


def coerce_value(value: Any) -> Any:
    """Рекурсивно приводит значение: скаляр, список/кортеж или словарь."""
    if isinstance(value, dict):
        return {k: coerce_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        coerced = [coerce_value(item) for item in value]
        return type(value)(coerced) if isinstance(value, tuple) else coerced
    return coerce_scalar(value)


def coerce_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Приводит типы всех значений в словаре kwargs (используется для параметров
    библиотечных функций построения графиков — bins, alpha, kde, bw_adjust и т.п.)."""
    return {key: coerce_value(val) for key, val in kwargs.items()}
