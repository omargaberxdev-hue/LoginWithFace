# logging.py
import logging
import structlog
from opentelemetry import trace

def _add_trace_context(logger, method_name, event_dict):
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict

_logger = None

def setup_logging(log_file: str = "logs/app.jsonl"):
    global _logger
    logging.basicConfig(filename=log_file, level=logging.INFO)
    structlog.configure(
        processors=[_add_trace_context, structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    _logger = structlog.get_logger()

def get_logger():
    if _logger is None:
        raise RuntimeError("setup_logging() must be called before get_logger()")
    return _logger