from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

# Полный список полей, которые Planner (или пользователь через overrides) может задать
# для ОДНОГО конкретного графика. В отличие от ThemeManager/DEFAULT_THEME (который задаёт
# общие для всего набора графиков вещи: dpi, размер фигуры, шрифт и т.п.), PlotDescription
# описывает внешний вид именно этого графика: подписи, цвет, аннотации.
PLOT_DESCRIPTION_KEYS = (
    "title",
    "xlabel",
    "ylabel",
    "color",
    "palette",
    "legend",
    "grid",
    "font_size",
    "annotations",
    "color_mapping",
)


@dataclass
class Annotation:
    text: str
    x: Optional[float] = None
    y: Optional[float] = None
    # Если x/y не заданы — текст кладётся в угол графика (относительные axes-координаты).
    position: str = "top_right"  # top_right | top_left | bottom_right | bottom_left | data

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан Annotation:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Annotation":
        return Annotation(
            text=data["text"],
            x=data.get("x"),
            y=data.get("y"),
            position=data.get("position", "top_right"),
        )


@dataclass
class PlotDescription:
    """Явное, типизированное описание внешнего вида ОДНОГО графика.

    Заполняется PlotDescriptionResolver-ом из ChartTask.style (+ ChartTask.semantic как
    fallback для title) и передаётся дальше в ChartExecutionPlan. Backend'ы (matplotlib_backend,
    seaborn_backend) обязаны явно применять КАЖДОЕ непустое поле отсюда — в отличие от
    текущей ситуации, где часть полей (title) применяется случайно, а часть (color,
    xlabel/ylabel текст, аннотации) не применяется вовсе.
    """

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    color: Optional[str] = None
    palette: Optional[str] = None
    legend: Optional[bool] = None
    grid: Optional[bool] = None
    font_size: Optional[int] = None
    annotations: List[Annotation] = field(default_factory=list)
    # Явное соответствие категория -> цвет (для hue-группировок), например
    # {"setosa": "pink", "versicolor": "yellow"}. Отличается от "color" (один цвет
    # на весь график) и "palette" (имя встроенной seaborn-палитры).
    color_mapping: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан PlotDescription:\n {pformat(self.to_dict())}")  #f-строка вычисляется в любом случае, поэтому нужна проверка или форматирование %s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PlotDescription":
        data = dict(data or {})
        raw_annotations = data.pop("annotations", []) or []
        annotations = [
            a if isinstance(a, Annotation) else Annotation.from_dict(a)
            for a in raw_annotations
        ]
        known = {k: v for k, v in data.items() if k in PLOT_DESCRIPTION_KEYS}
        return PlotDescription(annotations=annotations, **known)

    def merge(self, other: "PlotDescription") -> "PlotDescription":
        """Возвращает новое PlotDescription, где непустые поля other перекрывают self."""
        merged = self.to_dict()
        for k, v in other.to_dict().items():
            if v not in (None, [], "", {}):
                merged[k] = v
        return PlotDescription.from_dict(merged)
