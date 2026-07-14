from dataclasses import dataclass, asdict
from typing import Dict, Any
from models.dataset_context import DatasetContext

import logging
from pprint import pformat

logger = logging.getLogger(__name__)

@dataclass
class AnalysisInput:
    prompt: str
    dataset_context: DatasetContext

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" Создан AnalysisInput:\n {pformat(self.to_dict())}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "dataset_context": self.dataset_context.to_dict(),
        }
