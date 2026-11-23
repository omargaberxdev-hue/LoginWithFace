import atexit
import logging  # absolute import -> stdlib, not this file
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import SimpleQueue

import structlog
from opentelemetry import trace

from config.settings import settings

_configured = False


def add_otel_context(logger, method_name, event_dict):
    """Inject the active trace_id / span_id into every log line."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def setup_logging():
    global _configured
    if _configured:  # idempotent
        return
    _configured = True

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_otel_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    # BUG FIX: QueueHandler.prepare() formats the record in the CALLING thread and
    # flattens it to a string. The JSON formatter has to live on the QueueHandler
    # (request thread: contextvars + active span exist), not on the file handler.
    queue = SimpleQueue()
    queue_handler = QueueHandler(queue)
    queue_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)  # plain %(message)s = the JSON line

    listener = QueueListener(queue, file_handler, stdout_handler, respect_handler_level=True)
    listener.start()
    atexit.register(listener.stop)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # our request_completed log replaces uvicorn's access log
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False

    # echo=True used to dump every SQL statement (embedding vectors included) into logs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
