import os
import json
from models.visualization_report import VisualizationReport
from models.dataset_context import DatasetContext
from services.llm_client import LLMClient
from typing import Generator, Union

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "interpreter_prompt.txt")


class Interpreter:
    """LLM Agent. Интерпретирует VisualizationReport + DatasetContext и формирует ответ пользователю."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def run(self, report: VisualizationReport, dataset_context: DatasetContext, streaming : bool) -> Union[str, Generator[str, None, None]]:
        payload = {
            "report": report.to_dict(),
            "dataset_context": dataset_context.to_dict(),
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        if streaming:
            return self.llm_client.chat_stream(self.system_prompt, user_prompt, temperature=0.4)  #передаём следующей функции генератор
        else:
            return self.llm_client.chat(self.system_prompt, user_prompt, temperature=0.4) #передаём следующей функции строку
