import os
import json
import re
import yaml
from models.planner_input import PlannerInput
from models.visualization_task import VisualizationTask
from services.llm_client import LLMClient
from services.registry import Registry, REGISTRY_DIR

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "planner_prompt.txt")


def _build_registry_kwargs_section(registry: Registry) -> str:
    """Строит раздел промпта со списком библиотечных kwargs (defaults из registry/*.yaml)
    на каждый chart_type. Planner исторически ничего не знал об API библиотек и видел
    только semantic/PlotDescription поля — поэтому никогда не мог переопределить,
    например, "bins"/"kde" для гистограммы, т.к. просто не знал об их существовании.
    Генерируется динамически из registry, чтобы не расходиться с кодом при добавлении
    новых типов графиков или новых default-параметров."""
    lines = [
        "",
        "Помимо semantic и style(PlotDescription/theme), КАЖДЫЙ chart_type может иметь",
        "собственные библиотечные параметры (defaults) — их тоже можно переопределять",
        "через \"style\" тем же именем ключа, если пользователь просит изменить поведение",
        "конкретного графика (число бинов, показывать ли KDE, прозрачность и т.п.):",
        "",
    ]
    for chart_type in registry.available_types():
        spec = registry.get(chart_type)
        defaults = spec.get("defaults") or {}
        if not defaults:
            continue
        params = ", ".join(f'"{k}"={v!r}' for k, v in defaults.items())
        lines.append(f"- {chart_type} ({spec.get('backend')}.{spec.get('function')}): {params}")
    lines.append("")
    lines.append(
        "Значения выше — это ТЕКУЩИЕ default'ы библиотеки, а не обязательные значения. "
        "Если пользователь просит другое (например, \"30 бинов\", \"без KDE\", "
        "\"полупрозрачные точки\") — положи нужное значение в style тем же ключом."
    )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    # снимаем возможные markdown-обёртки ```json ... ```
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Planner вернул ответ без JSON-объекта:\n{text}")
    return json.loads(text[start:end + 1])


class Planner:
    """LLM Agent. Определяет какие графики нужны и с какой семантикой. О высокоуровневых
    "тюнинговых" библиотечных параметрах (bins/kde/alpha и т.п.) знает из динамически
    сгенерированного раздела промпта (см. _build_registry_kwargs_section) — про сам API
    библиотек (как их вызывать, какой backend) по-прежнему не знает, этим занимается
    APIAdapter/ParameterResolver."""

    def __init__(self, llm_client: LLMClient, registry: Registry = None):
        self.llm_client = llm_client
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            base_prompt = f.read()
        registry = registry or Registry()
        self.system_prompt = base_prompt + "\n" + _build_registry_kwargs_section(registry)

    def run(self, planner_input: PlannerInput) -> VisualizationTask:
        user_prompt = json.dumps(planner_input.to_dict(), ensure_ascii=False, indent=2)
        raw = self.llm_client.chat(self.system_prompt, user_prompt, temperature=0.2)
        data = _extract_json(raw)
        return VisualizationTask.from_dict(data)