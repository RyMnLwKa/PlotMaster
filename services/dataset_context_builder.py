import pandas as pd
import numpy as np
from models.dataset_context import DatasetContext


class DatasetContextBuilder:
    """Service. Формирует DatasetContext из pd.DataFrame. Не является LLM-агентом."""

    def run(self, df: pd.DataFrame, target: str = None) -> DatasetContext:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_columns = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

        null_counts = df.isnull().sum().to_dict()
        null_percent = (df.isnull().mean() * 100).round(2).to_dict()
        unique_counts = df.nunique(dropna=True).to_dict()

        sample = df.head(5).replace({np.nan: None}).to_dict(orient="records")

        column_info = {}
        for col in df.columns:
            info = {"dtype": str(df[col].dtype)}
            if col in numeric_columns:
                desc = df[col].describe()
                info.update({
                    "min": float(desc.get("min", np.nan)) if pd.notnull(desc.get("min")) else None,
                    "max": float(desc.get("max", np.nan)) if pd.notnull(desc.get("max")) else None,
                    "mean": float(desc.get("mean", np.nan)) if pd.notnull(desc.get("mean")) else None,
                    "std": float(desc.get("std", np.nan)) if pd.notnull(desc.get("std")) else None,
                })
            elif col in categorical_columns:
                top_values = df[col].value_counts().head(5).to_dict()
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            column_info[col] = info

        return DatasetContext(
            shape=list(df.shape),
            columns=df.columns.tolist(),
            dtypes={c: str(t) for c, t in df.dtypes.items()},
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            null_counts={k: int(v) for k, v in null_counts.items()},
            null_percent={k: float(v) for k, v in null_percent.items()},
            unique_counts={k: int(v) for k, v in unique_counts.items()},
            sample=sample,
            column_info=column_info,
            target=target,
        )
