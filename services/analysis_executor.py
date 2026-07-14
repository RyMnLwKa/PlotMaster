import builtins
import threading
from typing import List, Optional, Callable

import numpy as np
import pandas as pd

from models.analysis_task import AnalysisPlan, AnalysisRequest
from models.data_object import DataObject, to_error_object


class AnalysisExecutionError(Exception):
    pass


class _Timeout(Exception):
    pass


# Минимальный безопасный набор builtins, доступный коду анализа.
_SAFE_BUILTIN_NAMES = (
    "range", "len", "enumerate", "zip", "min", "max", "sum", "sorted",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "isinstance", "print", "abs", "round", "map", "filter", "any", "all",
    "reversed", "type", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "StopIteration", "True", "False", "None",
)

# Явно запрещённые builtins, даже если случайно попали бы в список выше.
_FORBIDDEN_BUILTIN_NAMES = ("open", "__import__", "eval", "exec", "compile",
                             "input", "exit", "quit", "globals", "locals",
                             "vars", "getattr", "setattr", "delattr",
                             "__build_class__")

# Белый список префиксов модулей, разрешённых к импорту.
_ALLOWED_IMPORT_PREFIXES = ("sklearn.", "sklearn", "scipy.", "scipy",
                             "statsmodels.", "statsmodels", "numpy", "numpy.",
                             "pandas", "pandas.")

# Явный чёрный список визуализационных библиотек. Формально они и так не проходят
# белый список выше, но перечисляем их отдельно и проверяем ПЕРВЫМИ, чтобы:
#   1) дать разработчику/аналитику однозначное, а не общее ImportError-сообщение
#      ("это про графики", а не "модуль не входит в белый список");
#   2) иметь defense-in-depth на случай будущих правок белого списка.
# Этап анализа отвечает только за вычисления и подготовку данных — построение
# графиков выполняется отдельным этапом визуализации, а не кодом Analyst'а.
_FORBIDDEN_VISUALIZATION_PREFIXES = (
    "matplotlib", "matplotlib.", "seaborn", "seaborn.", "plotly", "plotly.",
    "bokeh", "bokeh.", "altair", "altair.", "plotnine", "plotnine.",
    "pygal", "pygal.", "dash", "dash.", "pylab", "pylab.",
    "mpl_toolkits", "mpl_toolkits.", "graphviz", "graphviz.",
)


def _make_restricted_import():
    real_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == p or name.startswith(p) for p in _FORBIDDEN_VISUALIZATION_PREFIXES):
            raise ImportError(
                f"Импорт визуализационной библиотеки '{name}' запрещён на этапе анализа. "
                f"Analyst отвечает только за вычисления и подготовку данных — построение "
                f"графиков выполняется отдельным этапом визуализации."
            )
        if not any(name == p or name.startswith(p) for p in _ALLOWED_IMPORT_PREFIXES):
            raise ImportError(
                f"Импорт модуля '{name}' запрещён в песочнице анализа. "
                f"Разрешены только sklearn.*, scipy.*, statsmodels.*, numpy, pandas."
            )
        return real_import(name, globals, locals, fromlist, level)

    return restricted_import


def _build_safe_builtins():
    safe = {}
    for name in _SAFE_BUILTIN_NAMES:
        if name in _FORBIDDEN_BUILTIN_NAMES:
            continue
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    safe["__import__"] = _make_restricted_import()
    # Двойная защита: явно вычёркиваем запрещённые имена, даже если бы попали выше.
    for name in _FORBIDDEN_BUILTIN_NAMES:
        if name == "__import__":
            continue
        safe.pop(name, None)
    return safe


def _run_with_timeout(fn: Callable, timeout_seconds: float):
    """Исполняет fn() с ограничением по времени через отдельный поток (кроссплатформенно)."""
    result = {}
    error = {}

    def target():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - специально ловим всё
            error["exc"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise _Timeout(f"Исполнение шага превысило лимит времени {timeout_seconds} сек.")
    if "exc" in error:
        raise error["exc"]
    return result


class AnalysisExecutor:
    """
    Sandboxed runtime. Исполняет python-код, сгенерированный Analyst'ом
    (AnalysisRequest.code), и собирает "сырые" значения через emit(id, value).
    Метаданные результата (type/description/aligned_with_dataset/columns) в код
    НЕ передаются — они берутся из YAML-контракта (AnalysisRequest.outputs),
    после чего собираются полноценные DataObject.

    Безопасность:
      - урезанный __builtins__ (без open/__import__(прямого)/eval/exec/compile/...);
      - кастомный __import__, пропускающий только sklearn.*/scipy.*/statsmodels.*/
        numpy/pandas;
      - df передаётся как df.copy(), чтобы код анализа не мог испортить исходные данные;
      - таймаут исполнения (по умолчанию 30 секунд);
      - перехват всех исключений (включая таймаут и запрещённый импорт) без падения
        пайплайна — вместо этого возвращается error-DataObject;
      - один retry: если код упал, ошибка передаётся обратно в Analyst с просьбой
        исправить код, и делается ровно одна повторная попытка.
    """

    def __init__(self, timeout_seconds: float = 30.0, analyst=None):
        self.timeout_seconds = timeout_seconds
        # analyst опционален — используется только для retry (agents.analyst.Analyst,
        # у которого есть метод fix_code). Если не передан, retry просто не выполняется.
        self.analyst = analyst

    def run(self, plan: AnalysisPlan, df: pd.DataFrame, max_retries : int) -> List[DataObject]:
        objects: List[DataObject] = []
        for request in plan.requests:
            objects.extend(self._run_request(request, df, max_retries))
        return objects

    def _run_request(self, request: AnalysisRequest, df: pd.DataFrame, max_retries : int) -> List[DataObject]:

        current_code = request.code
        last_error : Exception | None = None

        for retry in range(1, max_retries+2): # 0 попытка + 1, 2, ..., max_retries справлений
            print(f"Попытка №{retry}")
            try:
                raw_values = self._exec_code(current_code, df)
                return self._materialize(request, raw_values)
            except Exception as exc:  # noqa: BLE001 - намеренно ловим всё

                last_error = exc

                condition = (retry < max_retries) and (self.analyst is not None) and (hasattr(self.analyst, "fix_code"))
                if not condition:
                    break

                try:
                    current_code = self.analyst.fix_code(
                    request.id, current_code, request.purpose, str(exc))
                except Exception as fix_exc:  # noqa: BLE001 - если сама LLM упала при ретрае
                    last_error = Exception(f"Возникла ошибка при исправлении кода LLM: {fix_exc}")
                    break

        return [to_error_object(
            obj_id=f"{request.id}_error",
            description=f"Ошибка выполнения анализа (После {max_retries} попыток)",
            error_message=str(last_error),
            source_step_id=request.id)]

    def _exec_code(self, code: str, df: pd.DataFrame) -> dict:
        """Исполняет код в песочнице, возвращает {id: value} из вызовов emit(id, value)."""
        collected: dict = {}

        def emit(id, value):
            collected[id] = value

        restricted_globals = {
            "__builtins__": _build_safe_builtins(),
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "emit": emit,
        }
        restricted_locals: dict = {}

        def target():
            exec(code, restricted_globals, restricted_locals)  # noqa: S102 - намеренно, песочница

        _run_with_timeout(target, self.timeout_seconds)
        return collected

    def _materialize(self, request: AnalysisRequest, raw_values: dict) -> List[DataObject]:
        """Собирает DataObject из YAML-декларации (request.outputs) + значений из emit()."""
        if not request.outputs:
            return [to_error_object(
                obj_id=f"{request.id}_error",
                description="Некорректный контракт анализа",
                error_message=(
                    f"У шага '{request.id}' не задан outputs-контракт (YAML) — "
                    f"неизвестно, каким должен быть результат emit()."
                ),
                source_step_id=request.id,
            )]

        results: List[DataObject] = []
        for spec in request.outputs:
            if spec.id not in raw_values:
                results.append(to_error_object(
                    obj_id=f"{spec.id}_error",
                    description="Некорректный результат анализа",
                    error_message=(
                        f"Код шага '{request.id}' не вызвал emit('{spec.id}', ...), "
                        f"хотя это заявлено в outputs-контракте."
                    ),
                    source_step_id=request.id,
                ))
                continue

            obj = DataObject(
                id=spec.id,
                type=spec.type,
                value=raw_values[spec.id],
                description=spec.description,
                aligned_with_dataset=spec.aligned_with_dataset,
                columns=list(spec.columns) if spec.columns is not None else None,
                metadata=spec.metadata or {},
                source_step_id=request.id,
            )
            error = obj.validate()
            if error:
                results.append(to_error_object(
                    obj_id=f"{spec.id}_error",
                    description="Некорректный результат анализа",
                    error_message=error,
                    source_step_id=request.id,
                ))
            else:
                results.append(obj)

        # emit() с id, не заявленным в outputs — тоже сигнал рассинхронизации контракта и кода.
        declared_ids = {spec.id for spec in request.outputs}
        for extra_id in set(raw_values) - declared_ids:
            results.append(to_error_object(
                obj_id=f"{extra_id}_error",
                description="Некорректный результат анализа",
                error_message=(
                    f"Код шага '{request.id}' вызвал emit('{extra_id}', ...), "
                    f"не заявленный в outputs-контракте."
                ),
                source_step_id=request.id,
            ))

        return results
