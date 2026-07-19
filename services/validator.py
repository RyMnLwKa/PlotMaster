import pandas as pd
from models.visualization_task import ChartTask


class ValidationError(Exception):
    pass


class Validator:
    """Внутренний Service Executor'а. Проверяет корректность ChartTask относительно DataFrame."""

    COLUMN_SEMANTIC_KEYS = ["x", "y", "color_by", "size_by", "hue", "column", "columns", "target",
                            "yerr_by", "xerr_by", "labels_by", "y_true", "y_pred",  "X_train", "y_train",
                            "X_test", "y_test"]

    def run(self, chart_task: ChartTask, df: pd.DataFrame) -> ChartTask:
        semantic = dict(chart_task.semantic or {})

        for key in self.COLUMN_SEMANTIC_KEYS:
            value = semantic.get(key)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for col in values:
                if col not in df.columns:
                    raise ValidationError(
                        f"[{chart_task.id}] Столбец '{col}' (параметр '{key}') "
                        f"отсутствует в датасете. Доступные столбцы: {list(df.columns)}"
                    )

        if len(df) == 0:
            raise ValidationError(f"[{chart_task.id}] DataFrame пуст, построение графика невозможно.")

        chart_task.semantic = semantic
        return chart_task
