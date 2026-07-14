from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class VisualizationArtifact:
    chart_id: str
    semantic: Dict[str, Any]
    image_path: str
    backend: str
    function: str
    execution_time: float
    status: str
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Создан VisualizationArtifact:\n {format(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
