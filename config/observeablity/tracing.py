# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import json
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class FileSpanExporter(SpanExporter):
    def __init__(self, path="logs/traces.jsonl"):
        self.path = path

    def export(self, spans):
        with open(self.path, "a") as f:
            for span in spans:
                f.write(json.dumps({
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "start": span.start_time,
                    "end": span.end_time,
                    "duration_ns": span.end_time - span.start_time,
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes),
                }) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

_tracer = None  # module-level, set once by setup_tracing()

def setup_tracing():
    """THIS is the missing connection. Without calling this, trace.get_tracer()
    anywhere else in your code returns a no-op tracer -- spans get created and
    silently thrown away, never reaching FileSpanExporter."""
    global _tracer
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter()))
    trace.set_tracer_provider(provider)   # <-- registers it as THE global provider
    _tracer = trace.get_tracer("liveness.pipeline")

def get_tracer():
    if _tracer is None:
        raise RuntimeError("setup_tracing() must be called before get_tracer()")
    return _tracer