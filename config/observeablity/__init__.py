import os

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config.settings import settings

from .logging import setup_logging

_done = False


def setup_observability():
    """Call once at import time in main.py, BEFORE the app is created."""
    global _done
    if _done:
        return
    _done = True

    resource = Resource.create({"service.name": settings.service_name})

    # tracing: spans come from the default FastAPI / SQLAlchemy instrumentation
    provider = TracerProvider(resource=resource)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # metrics: default HTTP/DB metrics, scraped at /metrics
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[PrometheusMetricReader()])
    )

    setup_logging()  # reads the active span lazily, so order vs tracing isn't critical


def instrument_app(app: FastAPI):
    """Must run before startup (middleware can't be added afterwards)."""
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics")


def instrument_engine(engine):
    SQLAlchemyInstrumentor().instrument(engine=engine)
