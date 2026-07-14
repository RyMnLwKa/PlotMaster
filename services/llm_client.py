"""
Единый клиент для доступа к LLM.

Поддерживает два бэкенда:
  - openrouter: модель 'openrouter/auto' через OpenRouter API (OpenAI-совместимый /chat/completions)
  - ollama: локальные модели через библиотеку ollama, список моделей берётся из `ollama list`

Это единственное место в проекте, где происходит обращение к LLM.
Agents (Planner, Interpreter) используют этот клиент, ничего не зная о деталях провайдера.
"""

import os
import json
import requests
import codecs
from typing import Optional, Generator


class LLMClient:
    def __init__(self, backend: str, model: str, api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        if backend == "openrouter" and not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Не найден OPENROUTER_API_KEY. Установите переменную окружения "
                    "или передайте ключ явно."
                )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        if self.backend == "openrouter":
            return self._chat_openrouter(system_prompt, user_prompt, temperature)
        elif self.backend == "ollama":
            return self._chat_ollama(system_prompt, user_prompt, temperature)
        else:
            raise ValueError(f"Неизвестный backend: {self.backend}")

    def chat_stream(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Generator[str, None, None]:
        if self.backend == "openrouter":
            yield from self._stream_openrouter(system_prompt, user_prompt, temperature)
        elif self.backend == "ollama":
            # Для Ollama в текущей реализации стриминг отключен, возвращаем строку как один чанк
            yield self._chat_ollama(system_prompt, user_prompt, temperature)
        else:
            raise ValueError(f"Неизвестный backend: {self.backend}")

    def _chat_openrouter(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _stream_openrouter(self, system_prompt: str, user_prompt: str, temperature: float) -> Generator[str, None, None]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True
        }

        decoder = codecs.getincrementaldecoder("utf-8")(errors='ignore')
        buffer = ""
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=180) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=False): #iter_lines
                if not chunk:
                    continue

                decoded_chunk = decoder.decode(chunk, final=False)
                buffer += decoded_chunk

                if len(buffer) > 1024 * 1024:  # 1 мб
                    last_newline = buffer.rfind('\n')
                    if last_newline != -1:
                        buffer = buffer[last_newline + 1:]  # Оставляем только последнюю неполную строку
                    else:
                        buffer = ""

                while True:
                    try:
                        line_end = buffer.find('\n')
                        if line_end == -1:
                            break

                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1:]

                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                decoder.decode(b'', final=True)
                                return
                            try:
                                data_obj = json.loads(data)
                                content = data_obj["choices"][0]["delta"].get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
                            except (KeyError, IndexError):
                                pass
                    except Exception:
                        break

    def _chat_ollama(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Библиотека 'ollama' не установлена. Установите её: pip install ollama"
            )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


def list_ollama_models():
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "Библиотека 'ollama' не установлена. Установите её: pip install ollama"
        )
    result = ollama.list()
    models = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
    names = []
    for m in models:
        if isinstance(m, dict):
            names.append(m.get("model") or m.get("name"))
        else:
            names.append(getattr(m, "model", None) or getattr(m, "name", None))
    return [n for n in names if n]