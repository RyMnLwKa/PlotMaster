import pandas as pd
from typing import List, Dict, Optional
from models.visualization_task import VisualizationTask, ChartTask
from models.execution_plan import ExecutionPlan
from services.validator import Validator, ValidationError
from services.api_adapter import APIAdapter
from services.parameter_resolver import ParameterResolver
from services.theme_manager import ThemeManager
from services.plot_description_resolver import PlotDescriptionResolver
from services.execution_plan_builder import ExecutionPlanBuilder
from services.registry import Registry


class Executor:
    """Центральный компонент: VisualizationTask + DataFrame -> ExecutionPlan.

    Помимо основного (дополненного результатами анализа) df, может принимать
    extra_frames — самостоятельные таблицы из DataObjectStore.standalone_frames()
    (например, таблицу model_comparison/feature_importance). Если chart_task.metadata
    ['source'] указывает на id такой таблицы, валидация и построение графика идут по ней.

    dicts — DataObjectStore.dicts(): {id: dict}, тот же уровень, что и extra_frames/
    model_source — самостоятельные DataObject'ы типа 'dict' (данные разной длины,
    которые нельзя было положить в DataFrame, например X_train/y_train/X_test/y_test
    для mlxtend plot_learning_curves). metadata['source'] может указывать и на них.
    В отличие от extra_frames, содержимое dict НЕ валидируется здесь по столбцам df —
    это делает сам backend (Tool резолвит dicts и передаёт значение backend'у напрямую),
    Executor лишь не должен ошибочно отвергать такой source как "не найденный".

    Если source не указан, но столбцы графика не найдены в основном df — Executor
    пытается автоматически найти единственную extra_frame, содержащую ВСЕ нужные
    столбцы, и использует её (подстраховка на случай, если Planner забыл указать source).
    """

    def __init__(self, output_dir: str = "output", theme_overrides: dict = None,
                 plot_description_overrides: dict = None):
        registry = Registry()
        self.validator = Validator()
        self.api_adapter = APIAdapter(registry)
        self.parameter_resolver = ParameterResolver()
        self.theme_manager = ThemeManager(theme_overrides)
        self.plot_description_resolver = PlotDescriptionResolver(plot_description_overrides)
        self.plan_builder = ExecutionPlanBuilder(output_dir)

    def _referenced_columns(self, chart_task: ChartTask) -> List[str]:
        cols = []
        for key in Validator.COLUMN_SEMANTIC_KEYS:
            value = chart_task.semantic.get(key)
            if value is None:
                continue
            cols.extend(value if isinstance(value, list) else [value])
        return cols

    def _autodetect_source(self, chart_task: ChartTask, extra_frames: Dict[str, pd.DataFrame]) -> Optional[str]:
        needed = set(self._referenced_columns(chart_task))
        if not needed:
            return None
        matches = [fid for fid, frame in extra_frames.items() if needed.issubset(set(frame.columns))]
        return matches[0] if len(matches) == 1 else None

    def run(self, task: VisualizationTask, df: pd.DataFrame,
            extra_frames: Optional[Dict[str, pd.DataFrame]] = None,
            dicts: Optional[Dict[str, Dict]] = None) -> ExecutionPlan:
        extra_frames = extra_frames or {}
        dicts = dicts or {}
        plan = ExecutionPlan(output_mode=task.output_mode, layout=task.layout)
        for chart_task in task.charts:
            source = (chart_task.metadata or {}).get("source")
            is_dict_source = source in dicts
            if source and source not in extra_frames and not is_dict_source:
                raise ValueError(
                    f"[{chart_task.id}] metadata.source='{source}' не найден среди результатов "
                    f"анализа: {list(extra_frames.keys()) + list(dicts.keys())}"
                )

            # dict-источник (например X_train/y_train/X_test/y_test для learning_curves)
            # не таблица — по её столбцам ничего проверять нельзя, поэтому валидируем/строим
            # план по основному df, а сам dict дальше резолвит и передаёт backend'у Tool,
            # так же, как и model_source.
            chart_df = extra_frames[source] if source and not is_dict_source else df

            if not source and extra_frames:
                # Planner мог сослаться на столбцы аналитической таблицы, забыв указать
                # metadata.source — пробуем автоматически найти подходящую таблицу перед
                # тем, как отдавать основной df (и, вероятно, падать с ValidationError).
                try:
                    self.validator.run(chart_task, df)
                except ValidationError:
                    detected = self._autodetect_source(chart_task, extra_frames)
                    if detected:
                        chart_df = extra_frames[detected]
                        chart_task.metadata = dict(chart_task.metadata or {})
                        chart_task.metadata["source"] = detected
                        chart_task.metadata["source_autodetected"] = True

            validated = self.validator.run(chart_task, chart_df)
            adapted = self.api_adapter.run(validated)
            resolved = self.parameter_resolver.run(adapted, validated)
            theme_style = dict(validated.style or {})
            # Planner иногда кладёт dpi/figsize в metadata.figure_settings вместо style
            # (воспринимая их как "пояснение", а не строгий параметр) — подстраховываемся,
            # чтобы явное пожелание пользователя по dpi/размеру фигуры не терялось молча.
            figure_settings = (validated.metadata or {}).get("figure_settings") or {}
            if "dpi" in figure_settings:
                theme_style.setdefault("dpi", figure_settings["dpi"])
            if "figsize" in figure_settings:
                theme_style.setdefault("figure_size", figure_settings["figsize"])
            if "figure_size" in figure_settings:
                theme_style.setdefault("figure_size", figure_settings["figure_size"])

            theme = self.theme_manager.run(theme_style)
            plot_description = self.plot_description_resolver.run(validated)
            chart_plan = self.plan_builder.run(validated, resolved, theme, plot_description)
            plan.charts.append(chart_plan)
        return plan