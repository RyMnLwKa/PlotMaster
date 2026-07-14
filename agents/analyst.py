import os
import re
import json
import yaml
from models.analysis_input import AnalysisInput
from models.analysis_task import AnalysisPlan
from services.llm_client import LLMClient

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "analyst_prompt.txt")

_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PY_BLOCK_RE = re.compile(r"```python(?:\s+id=(\S+))?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_yaml_and_code(text: str) -> dict:
    print(text)
    """
    Ответ Analyst'а — это YAML-контракт (requests[].id/purpose/outputs, metadata)
    в одном fenced-блоке ```yaml ... ``` и по одному fenced-блоку ```python ... ```
    (опционально с меткой `id=<request_id>`) для каждого request'а, в том же
    порядке, в котором requests перечислены в YAML.

    Такой формат не подвержен JSONDecodeError на длинном многострочном коде —
    код читается как есть, без экранирования переносов строк внутри JSON-строки.
    """
    yaml_matches = _YAML_BLOCK_RE.findall(text)
    if not yaml_matches:
        print("ОШИБКА")
        print("yaml", yaml_matches)
        code_blocks = _PY_BLOCK_RE.findall(text)  # список (id_or_None, code)
        print("code", code_blocks)
        raise ValueError(f"Analyst вернул ответ без YAML-блока ```yaml ... ```:\n{text}")
    contract = yaml.safe_load(yaml_matches[0]) or {}

    code_blocks = _PY_BLOCK_RE.findall(text)  # список (id_or_None, code)
    print("ПРАВИЛЬНО")
    print("yaml", yaml_matches)
    print("code", code_blocks)
    requests = contract.get("requests", []) or []

    # сопоставляем блоки кода с requests: сперва по явной метке id=..., остаток — по порядку
    by_id = {rid: code for rid, code in code_blocks if rid}
    remaining = [code for rid, code in code_blocks if not rid]
    remaining_iter = iter(remaining)

    for req in requests:
        req_id = req.get("id")
        if req_id in by_id:
            req["code"] = by_id[req_id].strip()
        else:
            try:
                req["code"] = next(remaining_iter).strip()
            except StopIteration:
                raise ValueError(
                    f"Для request '{req_id}' не найден соответствующий блок ```python ... ```:\n{text}"
                )

    contract["requests"] = requests
    return contract


class Analyst:
    """
    LLM Agent. Решает, нужен ли предварительный анализ данных ПЕРЕД построением
    визуализаций, и если нужен — сам пишет python-код для каждого шага (free-form
    code generation), формируя AnalysisPlan. Может вернуть пустой AnalysisPlan
    (requests=[]), если запрос пользователя не подразумевает анализа — в этом
    случае пайплайн просто пропускает этап анализа.

    Формат ответа LLM: YAML-контракт (что должно получиться и зачем) + отдельные
    python-блоки с кодом (как это реализовать) — см. prompts/analyst_prompt.txt.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def run(self, analysis_input: AnalysisInput) -> AnalysisPlan:
        user_prompt = json.dumps(analysis_input.to_dict(), ensure_ascii=False, indent=2)
        raw = self.llm_client.chat(self.system_prompt, user_prompt, temperature=0.2)
        data = _parse_yaml_and_code(raw)
        return AnalysisPlan.from_dict(data)

    def fix_code(self, request_id: str, code: str, purpose: str, error_message: str) -> str:
        """Просит Analyst исправить упавший код одного шага (используется для retry).
        Ответ — один блок ```python ... ```, без YAML и пояснений."""
        user_prompt = (
            f"Код шага анализа '{request_id}' упал с ошибкой при исполнении в песочнице.\n\n"
            f"purpose: {purpose}\n\n"
            f"Исходный код:\n```python\n{code}\n```\n\n"
            f"Текст ошибки:\n{error_message}\n\n"
            f"Исправь код с учётом ошибки и требований контракта (df, emit(id, value), "
            f"запрещённые импорты/builtins, id и значения emit должны совпадать с "
            f"исходным outputs-контрактом этого шага). Верни ТОЛЬКО ОДИН блок кода вида "
            f"```python\n<исправленный полный код>\n```, без YAML, без пояснений."
        )
        raw = self.llm_client.chat(self.system_prompt, user_prompt, temperature=0.2)
        matches = _PY_BLOCK_RE.findall(raw)
        if not matches:
            # запасной вариант: если модель не обернула ответ в ```python, берём как есть
            return raw.strip()
        return matches[0][1].strip()
