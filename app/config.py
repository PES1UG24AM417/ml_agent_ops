import os


def setting(name: str, default: str) -> str:
    return os.getenv(name, default)


REDIS_URL = setting("REDIS_URL", "redis://localhost:6379/0")
INTAKE_AGENT_URL = setting("INTAKE_AGENT_URL", "http://localhost:8101")
COVERAGE_AGENT_URL = setting("COVERAGE_AGENT_URL", "http://localhost:8102")
RATE_LIMIT_PER_MINUTE = int(setting("RATE_LIMIT_PER_MINUTE", "30"))
RUN_TIMEOUT_SECONDS = float(setting("RUN_TIMEOUT_SECONDS", "30"))
AGENT_HTTP_TIMEOUT_SECONDS = float(setting("AGENT_HTTP_TIMEOUT_SECONDS", "5"))
MAX_AGENT_RETRIES = int(setting("MAX_AGENT_RETRIES", "2"))
PROMPT_VERSION = setting("PROMPT_VERSION", "v2")
MODEL_STUB_DELAY_SECONDS = float(setting("MODEL_STUB_DELAY_SECONDS", "0.25"))
