import json
from pathlib import Path
from typing import Any

from app.agents.coverage import coverage_graph
from app.agents.intake import classify
from app.prompts import PROMPTS

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = ROOT / "evaluation" / "scenarios.json"


def build_fnol(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_number": "DEMO-1001",
        "loss_date": "2025-01-01",
        "vehicle_registration": "KA01AB1234",
        "description": scenario["description"],
        "claimed_amount": 25000,
        "vehicle_use": scenario["vehicle_use"],
    }


def evaluate_variant(prompt_version: str) -> dict[str, Any]:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    grounded = relevant = faithful = 0
    hallucinations = 0
    correct_outcomes = 0
    incorrect_outcomes = []
    for scenario in scenarios:
        fnol = build_fnol(scenario)
        classification = classify(fnol).model_dump(mode="json")
        result = coverage_graph.invoke(
            {
                "fnol": fnol,
                "classification": classification,
                "prompt_version": prompt_version,
            }
        )["answer"]
        passages = {item.clause_id: item.passage for item in result.retrieved_passages}
        citation_ids = {item.clause_id for item in result.citations}
        has_expected = scenario["supporting_passage"] in passages
        relevant += int(has_expected)
        faithful += int(bool(citation_ids) and citation_ids.issubset(passages))
        grounded += int(
            bool(citation_ids)
            and citation_ids.issubset(passages)
            and all(passages[item.clause_id] == item.passage for item in result.citations)
        )
        hallucinations += int(
            any(item.clause_id not in passages for item in result.citations)
        )
        outcome = "escalate" if result.needs_human_review else "recommend"
        if outcome == scenario["expected"]:
            correct_outcomes += 1
        else:
            incorrect_outcomes.append(scenario["id"])

    total = len(scenarios)
    return {
        "prompt_version": prompt_version,
        "prompt": PROMPTS[prompt_version],
        "scenarios": total,
        "expected_outcome_accuracy": round(correct_outcomes / total, 3),
        "incorrect_outcome_scenarios": incorrect_outcomes,
        "groundedness": round(grounded / total, 3),
        "context_relevance": round(relevant / total, 3),
        "faithfulness": round(faithful / total, 3),
        "hallucination_rate": round(hallucinations / total, 3),
        "metric_definitions": {
            "groundedness": "fraction of scenarios with a non-empty answer whose citations resolve verbatim to retrieved clauses",
            "context_relevance": "fraction where the expected supporting clause appears in the retrieved context",
            "faithfulness": "fraction where all returned citation IDs are present in retrieved context",
            "hallucination_rate": "fraction with any returned citation that is absent from retrieved context",
        },
    }


def main() -> None:
    report = {
        "baseline": evaluate_variant("v1"),
        "after_change": evaluate_variant("v2"),
        "change": "v2 adds a separate vehicle-use restriction/exclusion search before assessing coverage.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
