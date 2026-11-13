import json

from fastapi.testclient import TestClient

from app import api


class FakeRedis:
    def __init__(self):
        self.quarantined = []
        self.queue = []

    async def rpush(self, key, value):
        if key == "fnol:quarantine":
            self.quarantined.append(value)
        elif key == "fnol:queue":
            self.queue.append(value)

    async def aclose(self):
        pass


def test_invalid_fnol_is_quarantined_without_queueing(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(api, "redis_client", fake_redis)

    with TestClient(api.app) as client:
        response = client.post("/v1/claims", json={"policy_number": "bad"})

    assert response.status_code == 422
    assert response.json()["quarantined"] is True
    assert len(fake_redis.quarantined) == 1
    assert json.loads(fake_redis.quarantined[0])["errors"]
    assert fake_redis.queue == []
