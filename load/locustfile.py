import random
import time
from datetime import date
from uuid import uuid4

from locust import HttpUser, between, task


class ClaimsUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def submit_and_poll_claim(self):
        started = time.perf_counter()
        body = {
            "policy_number": "DEMO-1001",
            "loss_date": date.today().isoformat(),
            "vehicle_registration": f"KA01AB{random.randint(1000, 9999)}",
            "description": "My car was damaged in a collision on a personal trip.",
            "claimed_amount": 25000,
            "vehicle_use": "private",
        }
        with self.client.post(
            "/v1/claims",
            json=body,
            headers={"Idempotency-Key": str(uuid4())},
            name="POST /v1/claims",
            catch_response=True,
        ) as response:
            if response.status_code != 202:
                response.failure(f"submission returned {response.status_code}")
                return
            run_id = response.json()["run_id"]
            deadline = time.perf_counter() + 60
            while time.perf_counter() < deadline:
                with self.client.get(
                    f"/v1/runs/{run_id}", name="GET /v1/runs/{id}"
                ) as poll:
                    poll.raise_for_status()
                    result = poll.json()
                if result["status"] in {"completed", "escalated", "failed"}:
                    response.success()
                    self.environment.events.request.fire(
                        request_type="RUN",
                        name="POST submission to terminal result",
                        response_time=(time.perf_counter() - started) * 1000,
                        response_length=len(poll.content),
                        exception=None,
                        context={},
                    )
                    return
                time.sleep(0.1)
            response.failure("run did not reach a terminal state in 60 seconds")
