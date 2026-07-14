import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Создаем папку для логов
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Константы для форматирования
CONSOLE_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
FILE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

#logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False, log_file: str = "pipeline.log"):

    console_formatter = logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT)
    file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        LOG_DIR / log_file,
        maxBytes=10_000_000,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Максимальный уровень

    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    root_logger.info("Логирование настроено (verbose=%s)", verbose)
    root_logger.debug("Папка логов: %s", LOG_DIR.absolute())

    return root_logger