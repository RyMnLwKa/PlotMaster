from typing import Dict, Any

DEFAULT_THEME = {
    "style": "whitegrid",
    "palette": "deep",
    "dpi": 120,
    "figure_size": [8, 5],
    "font": "DejaVu Sans",
    "legend": True,
    "grid": True,
}


class ThemeManager:
    """Добавляет параметры оформления: style, palette, dpi, figure_size, font, legend, grid."""

    def __init__(self, overrides: Dict[str, Any] = None):
        self.overrides = overrides or {}

    def run(self, chart_task_style: Dict[str, Any] = None) -> Dict[str, Any]:
        theme = dict(DEFAULT_THEME)
        theme.update(self.overrides)
        if chart_task_style:
            for k in DEFAULT_THEME:
                if k in chart_task_style:
                    theme[k] = chart_task_style[k]
        return theme
