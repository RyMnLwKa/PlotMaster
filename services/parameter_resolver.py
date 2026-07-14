from typing import Dict, Any
from models.visualization_task import ChartTask
from models.plot_description import PLOT_DESCRIPTION_KEYS
from services.theme_manager import DEFAULT_THEME
from services.type_coercion import coerce_kwargs

# Ключи, которые принадлежат ThemeManager (figure-level: dpi, размер, шрифт, палитра
# по умолчанию и т.п.) — они ЯВЛЯЮТСЯ частью style, но НЕ являются kwargs библиотечной
# функции (sns.histplot(dpi=...) упадёт), поэтому их тоже нужно исключать из kwargs,
# как и поля PlotDescription.
THEME_KEYS = set(DEFAULT_THEME.keys())
NON_KWARG_STYLE_KEYS = set(PLOT_DESCRIPTION_KEYS) | THEME_KEYS


class ParameterResolver:
    """Подставляет параметры по умолчанию, пользовательские и библиотечные, формирует финальный kwargs."""

    def run(self, adapted: Dict[str, Any], chart_task: ChartTask) -> Dict[str, Any]:
        kwargs = dict(adapted["kwargs"])
        semantic_keys = adapted.get("semantic_keys", set())

        # style, заданные Planner'ом (chart_task.style), имеют приоритет над registry-
        # defaults (bins/kde/alpha и т.п. из yaml) — они ДОЛЖНЫ перезаписываться пользователем.
        # Но не над явными data-семантик-параметрами (x/y/hue), уже примененными в APIAdapter —
        # их style трогать не должен.
        # Поля PlotDescription (title/xlabel/ylabel/color/palette/legend/grid/font_size/
        # annotations/color_mapping) и поля ThemeManager (dpi/figure_size/style/font/...)
        # НЕ являются kwargs библиотечной функции — они применяются отдельно
        # (PlotDescriptionResolver -> backend.render, ThemeManager -> Figure/rcParams),
        # поэтому сюда не подмешиваются (иначе, например, sns.histplot(dpi=..., title=...)
        # упадёт с ошибкой неизвестного аргумента).
        for k, v in (chart_task.style or {}).items():

            if k in NON_KWARG_STYLE_KEYS:  #исключает системные параметры (темы) и общие параметры графика (plot), остаются только для самого графика (chart)
                continue
            if k in semantic_keys:
                kwargs.setdefault(k, v)
            else:
                kwargs[k] = v

        # Строгая типизация: значения kwargs могли прийти строками из YAML/JSON
        # (Planner, registry-defaults, пользовательский style) — например
        # bins: "30" вместо 30, из-за чего seaborn/matplotlib падают с ошибками
        # вида "bins must be an integer, a string, or an array". Так как заранее
        # неизвестен полный набор параметров каждой библиотечной функции, типы
        # приводятся не по жёсткой Pydantic-схеме "поле -> тип", а каскадно —
        # bool -> int -> float -> str (см. services/type_coercion.py).
        adapted["kwargs"] = coerce_kwargs(kwargs)
        return adapted
