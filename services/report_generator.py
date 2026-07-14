from typing import List, Dict, Any, Optional
from models.visualization_artifact import VisualizationArtifact
from models.visualization_report import VisualizationReport


class ReportGenerator:
    """Собирает список VisualizationArtifact (+ опционально метрики анализа) в единый VisualizationReport."""

    def run(self, original_prompt: str, artifacts: List[VisualizationArtifact],
            analysis_summary: Optional[Dict[str, Any]] = None) -> VisualizationReport:
        ok = [a for a in artifacts if a.status == "ok"]
        failed = [a for a in artifacts if a.status != "ok"]
        summary = {
            "total_charts": len(artifacts),
            "successful": len(ok),
            "failed": len(failed),
            "failed_ids": [a.chart_id for a in failed],
        }
        return VisualizationReport(
            original_prompt=original_prompt,
            artifacts=artifacts,
            summary=summary,
            analysis_summary=analysis_summary or {},
        )
