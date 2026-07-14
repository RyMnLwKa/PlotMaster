"""
Линейный конвейер, аналогичный Runnable Pipeline в LangChain, но без внешних зависимостей.

Расширенная схема (с опциональным этапом анализа):

    DataFrame -> DatasetContextBuilder -> DatasetContext
    DatasetContext + prompt -> AnalysisInput -> Analyst (LLM, free-form code) -> AnalysisPlan
    AnalysisPlan + DataFrame -> AnalysisExecutor (sandboxed runtime) -> DataObject[] -> DataObjectStore
    DataObjectStore.augmented_dataframe(df) -> df с добавленными "аналитическими" столбцами
    DatasetContext + prompt + DataObjectStore.context_summary() -> PlannerInput
                -> Planner (LLM) -> VisualizationTask
    VisualizationTask + augmented_df (+ DataObjectStore.standalone_frames) -> Executor -> ExecutionPlan
    ExecutionPlan + augmented_df (+ standalone_frames) -> Tool -> VisualizationArtifact[]
    VisualizationArtifact[] + DataObjectStore.metrics_summary() -> ReportGenerator -> VisualizationReport
    VisualizationReport + DatasetContext -> Interpreter (LLM) -> ответ пользователю
"""
import pandas as pd
from models.planner_input import PlannerInput
from models.analysis_input import AnalysisInput
from services.dataset_context_builder import DatasetContextBuilder
from services.analysis_executor import AnalysisExecutor
from services.data_object_store import DataObjectStore
from services.executor import Executor
from services.report_generator import ReportGenerator
from tools.tool import Tool
from agents.analyst import Analyst
from agents.planner import Planner
from agents.interpreter import Interpreter


class VisualizationPipeline:
    def __init__(self, planner: Planner, interpreter: Interpreter, analyst: Analyst = None,
                 output_dir: str = "output"):
        self.dataset_context_builder = DatasetContextBuilder()
        self.analyst = analyst
        self.analysis_executor = AnalysisExecutor(analyst=analyst)
        self.executor = Executor(output_dir=output_dir)
        self.tool = Tool()
        self.report_generator = ReportGenerator()
        self.planner = planner
        self.interpreter = interpreter

    def run(self, df: pd.DataFrame, prompt: str, target: str = None,
            run_analysis: bool = True, streaming : bool = False, verbose: bool = True, max_retries: int = 5):
        dataset_context = self.dataset_context_builder.run(df, target=target)
        if verbose:
            print("[1/8] DatasetContext построен.")

        store = DataObjectStore()
        if run_analysis and self.analyst is not None:
            analysis_input = AnalysisInput(prompt=prompt, dataset_context=dataset_context)
            analysis_plan = self.analyst.run(analysis_input)
            if verbose:
                print(f"[2/8] Analyst вернул {len(analysis_plan.requests)} шаг(ов) анализа.")

            if analysis_plan.requests:
                data_objects = self.analysis_executor.run(analysis_plan, df, max_retries)
                store.add_all(data_objects)
                if verbose:
                    print(f"[3/8] AnalysisExecutor выполнил анализ: {len(data_objects)} DataObject(ов).")
            else:
                if verbose:
                    print("[3/8] Анализ не требуется — этап пропущен.")
        else:
            if verbose:
                print("[2-3/8] Этап анализа отключён (run_analysis=False).")

        working_df = store.augmented_dataframe(df) if not store.is_empty() else df
        extra_frames = store.standalone_frames(base_len=len(df)) if not store.is_empty() else {}
        models = store.models() if not store.is_empty() else {}

        planner_input = PlannerInput(
            prompt=prompt,
            dataset_context=dataset_context,
            analysis_context=store.context_summary(base_len=len(df)),
        )
        task = self.planner.run(planner_input)
        if verbose:
            print(f"[4/8] Planner вернул задачу: {len(task.charts)} график(ов).")

        plan = self.executor.run(task, working_df, extra_frames=extra_frames)
        if verbose:
            print("[5/8] ExecutionPlan построен.")

        artifacts = self.tool.run(plan, working_df, extra_frames=extra_frames, models=models)
        if verbose:
            ok = sum(1 for a in artifacts if a.status == "ok")
            print(f"[6/8] Tool выполнил построение: {ok}/{len(artifacts)} успешно.")

        report = self.report_generator.run(prompt, artifacts, analysis_summary=store.metrics_summary())
        if verbose:
            print("[7/8] VisualizationReport сформирован.")

        answer = self.interpreter.run(report, dataset_context, streaming=streaming)  #передаём генератор или строку
        if verbose:
            print("[8/8] Interpreter сформировал ответ.\n")

        return report, answer
