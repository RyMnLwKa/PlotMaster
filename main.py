#!/usr/bin/env python3
"""
CLI-агент мультиагентной системы визуализации данных.

Использование:
    python main.py --data data.csv
    python main.py --data data.csv --backend openrouter --model openrouter/auto
    python main.py --data data.csv --backend ollama --model llama3.1
    python main.py --data data.csv --prompt "Покажи зависимость между X и Y" --target label

Если --backend не указан, CLI спросит интерактивно:
  1) OpenRouter ('openrouter/auto')
  2) Ollama (локальный список моделей через `ollama list`)
"""

import argparse
import os
import sys
import re
import pandas as pd
import logging

from services.logger_setup import setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.llm_client import LLMClient, list_ollama_models
from agents.analyst import Analyst
from agents.planner import Planner
from agents.interpreter import Interpreter
from pipeline import VisualizationPipeline

def clean_prompt(text: str) -> str:
    if not text:
        return text
    try:
        return ''.join(c for c in text if ord(c) < 0xD800 or ord(c) > 0xDFFF)
    except:
        return text.encode('utf-8', errors='replace').decode('utf-8')

def choose_backend_interactively():
    print("Выберите провайдера LLM:")
    print("  1) OpenRouter (модель: openrouter/auto)")
    print("  2) Ollama (локальные модели)")
    choice = input("Ваш выбор [1/2]: ").strip()

    if choice == "1":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            api_key = input("Введите OPENROUTER_API_KEY: ").strip()
        return "openrouter", "openrouter/auto", api_key

    elif choice == "2":
        try:
            models = list_ollama_models()
        except Exception as e:
            print(f"Не удалось получить список моделей ollama: {e}")
            sys.exit(1)
        if not models:
            print("Локальные модели ollama не найдены. Выполните `ollama pull <model>`.")
            sys.exit(1)
        print("Доступные локальные модели ollama:")
        for i, m in enumerate(models, 1):
            print(f"  {i}) {m}")
        idx = input(f"Выберите модель [1-{len(models)}]: ").strip()
        try:
            model = models[int(idx) - 1]
        except (ValueError, IndexError):
            print("Некорректный выбор.")
            sys.exit(1)
        return "ollama", model, None

    else:
        print("Некорректный выбор.")
        sys.exit(1)


def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Мультиагентная система визуализации данных (CLI)")
    parser.add_argument("--data", required=True, help="Путь к файлу с данными (csv/xlsx/json/parquet)")
    parser.add_argument("--prompt", help="Запрос на естественном языке для построения визуализаций")
    parser.add_argument("--target", default=None, help="Название целевого столбца (опционально)")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default=None,
                         help="Провайдер LLM. Если не указан — спросит интерактивно.")
    parser.add_argument("--model", default=None, help="Название модели (по умолчанию: openrouter/auto для openrouter)")
    parser.add_argument("--api-key", default=None, help="OPENROUTER_API_KEY (если backend=openrouter)")
    parser.add_argument("--output-dir", default="output", help="Папка для сохранения графиков")
    parser.add_argument("--no-analysis", action="store_true",
                         help="Отключить этап предварительного анализа (Analyst/AnalysisPlan)")
    parser.add_argument("-s", "--streaming", default=False, action="store_true", help="Выводить токены по мере генерации")
    parser.add_argument("--verbose", default=False, action="store_true", help="Включить DEBUG-логи")
    parser.add_argument("----max-analysis-retries", type=int, default=5, help="Максимальное количество попыток исправления кода анализа через LLM (default: 5)")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger.info("Запуск CLI-агента визуализации данных")
    logger.debug("Аргументы: %s", args)

    if not os.path.exists(args.data):
        logger.critical(f"Файл не найден: {args.data}")
        sys.exit(1)

    try:
        df = load_dataframe(args.data)

        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target

        logging.info(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    except Exception as e:
        logger.critical("Ошибка загрузки данных: %s", e, exc_info=True)
        sys.exit(1)

    if args.backend:
        backend = args.backend
        if backend == "openrouter":
            model = args.model or "openrouter/auto"
            api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        else:
            model = args.model
            if not model:
                try:
                    models = list_ollama_models()
                except Exception as e:
                    logger.error(f"Не удалось получить список моделей ollama: {e}")
                    sys.exit(1)
                if not models:
                    logger.error("Локальные модели ollama не найдены.")
                    sys.exit(1)
                logging.info("Доступные локальные модели ollama:")
                for i, m in enumerate(models, 1):
                    logging.info(f"  {i}) {m}")
                idx = input(f"Выберите модель [1-{len(models)}]: ").strip()
                model = models[int(idx) - 1]
            api_key = None
    else:
        backend, model, api_key = choose_backend_interactively()

    llm_client = LLMClient(backend=backend, model=model, api_key=api_key)

    prompt = args.prompt
    if not prompt:
        logger.debug("Промпт не указан, запрашиваем у пользователя")
        from prompt_toolkit import prompt
        prompt = prompt("\nВведите запрос для визуализации данных: ").strip()
        #prompt = input("\nВведите запрос для визуализации данных: ").strip()

    prompt = clean_prompt(prompt)
    logger.debug("Промпт: %s", prompt)

    analyst = Analyst(llm_client)
    planner = Planner(llm_client)
    interpreter = Interpreter(llm_client)
    pipeline = VisualizationPipeline(planner, interpreter, analyst=analyst, output_dir=args.output_dir)

    print()

    logger.info("Запуск пайплайна...")
    logger.debug("Параметры: target=%s, streaming=%s, run_analysis=%s",
                 args.target, args.streaming, not args.no_analysis)

    report, answer = pipeline.run(
        df, prompt,
        target=args.target,
        streaming=args.streaming,
        run_analysis=not args.no_analysis,
        verbose=args.verbose,
        max_retries=args.max_analysis_retries)

    logger.info("Пайплайн завершен. Создано %s графиков", len(report.artifacts))
    logger.debug("Отчет: %s", report)

    print()
    print("=" * 70)
    print("ОТВЕТ:")

    if args.streaming:
        #Generator
        for token in answer:
            print(token, end="", flush=True)
        print()
    else:
        #str или одиночный генератор (случайно)
        full_answer = "".join(answer)
        print(full_answer)  #full_answer = next(answer)
        logger.debug("Длина ответа: %s символов", len(full_answer))

    print("=" * 70)
    print()

    print("Сохранённые графики:")
    for artifact in report.artifacts:
        status_mark = "✓" if artifact.status == "ok" else "✗"
        print(f"  [{status_mark}] {artifact.chart_id}: {artifact.image_path or artifact.warnings}")
        logger.debug("Артефакт %s: status=%s, path=%s",
                     artifact.chart_id, artifact.status, artifact.image_path)

    if report.analysis_summary:
        print("\nМетрики этапа анализа:")
        for step_id, data in report.analysis_summary.items():
            print(f"  - {step_id}: {data['value']}")
        logger.info("Метрики анализа: %s", report.analysis_summary)


if __name__ == "__main__":
    main()
