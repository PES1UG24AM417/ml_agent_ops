import logging
import os
from threading import Lock

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False
_lock = Lock()


def configure_tracing(service_name: str):
    global _configured
    with _lock:
        if not _configured:
            provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if endpoint:
                provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=endpoint,
                            insecure=True,
                        )
                    )
                )
            else:
                logging.getLogger(__name__).info(
                    "OpenTelemetry export disabled; OTEL_EXPORTER_OTLP_ENDPOINT is not set"
                )
            trace.set_tracer_provider(provider)
            _configured = True
    return trace.get_tracer(service_name)


def inject_trace_context(headers: dict[str, str]) -> None:
    propagate.inject(headers)
