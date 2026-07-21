# __init__.py
from .tracing import setup_tracing, get_tracer, FileSpanExporter
from .metrics import setup_metrics, record_duration
from .logging import setup_logging, get_logger
from .decorators import trace_stage, trace_pipeline

def setup_observability():
    setup_tracing()   # order matters: tracing before logging,
    setup_logging()    # so _add_trace_context has a real provider to read from
    setup_metrics()