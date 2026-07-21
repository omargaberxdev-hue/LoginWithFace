import time
import functools
import numpy as np
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Histogram, Counter

tracer = trace.get_tracer("liveness.pipeline")

# per-stage: labeled by stage name so one dashboard panel covers crop/hann/lbp/fft/embed
STAGE_DURATION = Histogram("pipeline_stage_duration_seconds", "Stage duration", ["stage"])
STAGE_ERRORS = Counter("pipeline_stage_errors_total", "Stage errors", ["stage"])

# end-to-end: labeled by flow (signup/signin) so you can compare them separately
PIPELINE_DURATION = Histogram("auth_pipeline_duration_seconds", "End-to-end duration", ["flow"])
PIPELINE_REQUESTS = Counter("auth_pipeline_requests_total", "Requests by outcome", ["flow", "outcome"])

# domain-specific -- not generic, so kept explicit rather than inferred by a decorator
LIVENESS_RESULT = Counter("liveness_result_total", "Liveness verdicts", ["label"])
LIVENESS_CONFIDENCE = Histogram("liveness_confidence", "Model confidence", buckets=[i / 10 for i in range(11)])


def _add_trace_context(logger, method_name, event_dict):
    """Injects trace_id/span_id into every log line -- this is what lets you
    jump from a Loki log line straight to the matching Tempo trace."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


structlog.configure(processors=[
    _add_trace_context,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()


def _array_attrs(prefix, arr):
    """Shape/dtype only -- never the raw pixel/feature array itself."""
    if isinstance(arr, np.ndarray):
        return {f"{prefix}.shape": str(arr.shape), f"{prefix}.dtype": str(arr.dtype)}
    return {}


def trace_stage(stage_name: str, expected_exceptions: tuple = ()):
    """
    For internal pipeline steps (crop, hann window, LBP, FFT, embed).
    Child span + per-stage metric, no log line on success -- the span
    already carries timing/shape, a log here would just duplicate it.

    expected_exceptions: business-rule rejections (e.g. "no face found")
    that aren't system errors -- these get a WARN log, not ERROR, and
    don't count toward STAGE_ERRORS.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
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
                    STAGE_ERRORS.labels(stage=stage_name).inc()
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log.error("stage_failed", stage=stage_name, error=str(exc))
                    raise
                finally:
                    STAGE_DURATION.labels(stage=stage_name).observe(time.perf_counter() - start)
        return wrapper
    return decorator


def trace_pipeline(flow_name: str):
    """For root entry points (sign_up, sign_in). Root span + entry/exit
    log + end-to-end duration + outcome counter."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"pipeline.{flow_name}") as span:
                log.info("pipeline_started", flow=flow_name)
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    PIPELINE_REQUESTS.labels(flow=flow_name, outcome="success").inc()
                    log.info("pipeline_completed", flow=flow_name)
                    return result
                except Exception as exc:
                    outcome = type(exc).__name__
                    PIPELINE_REQUESTS.labels(flow=flow_name, outcome=outcome).inc()
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log.warning("pipeline_rejected", flow=flow_name, reason=outcome)
                    raise
                finally:
                    PIPELINE_DURATION.labels(flow=flow_name).observe(time.perf_counter() - start)
        return wrapper
    return decorator