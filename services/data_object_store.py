import pandas as pd
from typing import List, Dict, Any, Optional
from models.data_object import DataObject


class DataObjectStore:
    """
    Хранилище DataObject'ов, порождённых AnalysisExecutor'ом в рамках одного запроса.

    Не является ни LLM Agent, ни частью основного конвейера Executor/Tool —
    это промежуточный "мост" между AnalysisPlan и VisualizationTask:

      1. AnalysisExecutor кладёт сюда результаты (PCA-компоненты, метки кластеров,
         таблицу feature importance, метрики моделей и т.д.).
      2. Store умеет "материализовать" объекты типа series/dataframe в копию
         исходного DataFrame — после этого Executor/Validator визуализации видят
         их как обычные столбцы и могут строить графики (scatter по PCA-компонентам,
         barplot по feature importance, color_by=метка кластера и т.д.) без каких-либо
         изменений в коде визуализации.
      3. Store формирует компактную сводку (context_summary) для Planner и
         Interpreter — что доступно и что это значит.
    """

    def __init__(self):
        self._objects: Dict[str, DataObject] = {}

    def add(self, obj: DataObject) -> None:
        self._objects[obj.id] = obj

    def add_all(self, objs: List[DataObject]) -> None:
        for obj in objs:
            self.add(obj)

    def get(self, id: str) -> Optional[DataObject]:
        return self._objects.get(id)

    def all(self) -> List[DataObject]:
        return list(self._objects.values())

    def is_empty(self) -> bool:
        return len(self._objects) == 0

    def augmented_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает копию df с добавленными столбцами из series/dataframe DataObject'ов,
        выровненных по строкам исходного датасета (obj.aligned_with_dataset is True)."""
        result = df.copy()

        for obj in self._objects.values():
            if obj.status == "error":
                continue

            if obj.type == "series":
                if not obj.aligned_with_dataset or len(obj.value) != len(result):
                    continue  # не выровнено по строкам df — пропускаем
                col_name = obj.id if obj.id not in result.columns else f"{obj.id}_analysis"
                result[col_name] = pd.Series(obj.value).reset_index(drop=True).set_axis(result.index)

            elif obj.type == "dataframe":
                value_df = obj.value
                if not obj.aligned_with_dataset or len(value_df) != len(result):
                    # не выровнено по строкам исходного df (например, сводная таблица
                    # сравнения моделей с одной строкой на модель) — такие данные
                    # используются как самостоятельный DataFrame, см. standalone_frames()
                    continue
                for col in value_df.columns:
                    col_name = col if col not in result.columns else f"{obj.id}_{col}"
                    aligned = pd.Series(value_df[col]).reset_index(drop=True).set_axis(result.index)
                    result[col_name] = aligned

            # metrics / model — не материализуются в столбцы

        return result

    def models(self) -> Dict[str, Any]:
        """
        Возвращает DataObject'ы типа 'model' как {id: обученная модель}. Раньше модели,
        переданные Analyst'ом, использовались ТОЛЬКО для текстовой интерпретации
        (Interpreter). Теперь backend'ы визуализации, которым нужна уже обученная модель
        (например mlxtend decision_regions), тоже могут её получить — но НЕ обучают
        сами: Tool резолвит chart_task.metadata['model_source'] через этот метод и
        передаёт готовый объект модели в backend.run()/render(model=...).
        """
        models = {}
        for obj in self._objects.values():
            if obj.status == "error":
                continue
            if obj.type == "model":
                models[obj.id] = obj.value
        return models

    def standalone_frames(self, base_len: int = None) -> Dict[str, pd.DataFrame]:
        """
        Возвращает DataFrame-объекты, НЕ выровненные по строкам исходного датасета
        (obj.aligned_with_dataset is False) — например, таблица сравнения моделей:
        одна строка на модель. Такие DataObject'ы нужно построить как самостоятельный
        график (chart_task.metadata.source = id), а не примешивать к основному df
        через augmented_dataframe.
        """
        frames = {}
        for obj in self._objects.values():
            if obj.status == "error":
                continue
            if obj.type == "dataframe" and not obj.aligned_with_dataset:
                frames[obj.id] = obj.value
        return frames

    def context_summary(self, base_len: int = None) -> List[Dict[str, Any]]:
        summary = []
        for obj in self._objects.values():
            entry = {
                "id": obj.id,
                "type": obj.type,
                "metadata": obj.metadata,
                "description": obj.description,
                "status": obj.status,
            }
            if obj.status == "error":
                entry["usable_as_column"] = False
                entry["hint"] = f"Шаг анализа завершился ошибкой: {'; '.join(obj.warnings)}"
                summary.append(entry)
                continue

            if obj.type == "series":
                entry["usable_as_column"] = True
                entry["aligned_with_dataset"] = obj.aligned_with_dataset
                entry["hint"] = f"{obj.description} Доступен как обычный столбец с именем '{obj.id}'."
            elif obj.type == "dataframe":
                cols = obj.columns or list(obj.value.columns)
                entry["usable_as_column"] = True
                entry["aligned_with_dataset"] = obj.aligned_with_dataset
                entry["columns"] = cols
                if obj.aligned_with_dataset:
                    entry["hint"] = (
                        f"{obj.description} Столбцы {cols} выровнены по исходным строкам — используй их "
                        f"напрямую как x/y/color_by наравне с обычными столбцами (например x='{cols[0]}')."
                    )
                else:
                    entry["hint"] = (
                        f"{obj.description} Самостоятельная таблица (не выровнена по строкам исходного "
                        f"датасета), столбцы {cols}. Для графика по этим данным укажи в metadata графика "
                        f"\"source\": \"{obj.id}\" — Executor построит его поверх этой таблицы "
                        f"вместо основного датасета (например barplot x='{cols[0]}', y='{cols[1]}')."
                    )
            elif obj.type == "model":
                entry["usable_as_column"] = False
                entry["hint"] = (
                    f"{obj.description} Обученная модель — не столбец графика, но может "
                    f"использоваться визуализацией, которой нужна уже обученная модель "
                    f"(например decision_regions). Если нужно построить такой график по "
                    f"этой модели, укажи в metadata графика \"model_source\": \"{obj.id}\" "
                    f"(в дополнение к обычным semantic.x/y/target) — Tool передаст готовую "
                    f"модель в backend, backend её НЕ переобучает."
                )
            else:
                entry["usable_as_column"] = False
                entry["hint"] = f"{obj.description} Метрики — использовать только в текстовой интерпретации, не как столбец графика."
            summary.append(entry)
        return summary

    def metrics_summary(self) -> Dict[str, Any]:
        """Только metrics/model объекты — для передачи Interpreter'у как численные факты."""
        return {
            obj.id: {"metadata": obj.metadata, "value": obj.value}
            for obj in self._objects.values()
            if obj.type == "metrics"
        }
