import sys
from pathlib import Path

import redis

from app.config import REDIS_URL


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
    client.ping()
elif not Path("/proc/1/cmdline").exists():
    raise SystemExit(1)
