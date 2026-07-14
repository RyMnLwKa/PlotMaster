from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from models.visualization_artifact import VisualizationArtifact

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class VisualizationReport:
    original_prompt: str
    artifacts: List[VisualizationArtifact] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    # Метрики и результаты этапа AnalysisPlan (PCA explained_variance, кластерная
    # inertia, R2 регрессии, feature importance и т.д.) — для текстовой интерпретации.
    # Пусто, если AnalysisPlan не выполнялся.
    analysis_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан VisualizationReport:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "summary": self.summary,
            "analysis_summary": self.analysis_summary,
        }
