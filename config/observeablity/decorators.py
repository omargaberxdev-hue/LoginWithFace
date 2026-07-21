# decorators.py
import time
import functools
import numpy as np
from opentelemetry.trace import Status, StatusCode
from .tracing import get_tracer
from .logging import get_logger
from .metrics import record_duration

def _array_attrs(prefix, arr):
    if isinstance(arr, np.ndarray):
        return {f"{prefix}.shape": str(arr.shape), f"{prefix}.dtype": str(arr.dtype)}
    return {}

def trace_stage(stage_name: str, expected_exceptions: tuple = ()):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()   # fetched at call time, not import time
            log = get_logger()      # same -- see explanation below
            with tracer.start_as_current_span(stage_name) as span:
                candidate = args[0] if args else next(iter(kwargs.values()), None)
                for k, v in _array_attrs("input", candidate).items():
                    span.set_attribute(k, v)
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    for k, v in _array_attrs("output", result).items():
                        span.set_attribute(k, v)
                    return result
                except expected_exceptions as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log.warning("stage_rejected", stage=stage_name, reason=str(exc))
                    raise
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log.error("stage_failed", stage=stage_name, error=str(exc))
                    raise
                finally:
                    record_duration(stage_name, time.perf_counter() - start)
        return wrapper
    return decorator

def trace_pipeline(flow_name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            log = get_logger()
            with tracer.start_as_current_span(f"pipeline.{flow_name}") as span:
                log.info("pipeline_started", flow=flow_name)
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    log.info("pipeline_completed", flow=flow_name)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log.warning("pipeline_rejected", flow=flow_name, reason=type(exc).__name__)
                    raise
                finally:
                    record_duration(f"pipeline.{flow_name}", time.perf_counter() - start)
        return wrapper
    return decorator