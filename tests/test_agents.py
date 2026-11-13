import json
from pathlib import Path

from app.agents.coverage import coverage_graph
from app.agents.intake import classify
from app.models import CoverageAssessment
from app.worker import verify_assessment


def scenario_fnol(description, vehicle_use="private"):
    return {
        "policy_number": "DEMO-1001",
        "loss_date": "2025-01-01",
        "vehicle_registration": "KA01AB1234",
        "description": description,
        "claimed_amount": 25000,
        "vehicle_use": vehicle_use,
    }


def test_food_delivery_case_retrieves_restriction_and_escalates():
    fnol = scenario_fnol(
        "I was making a food delivery run in my own car when the accident happened."
    )
    classification = classify(fnol).model_dump(mode="json")
    result = coverage_graph.invoke(
        {"fnol": fnol, "classification": classification, "prompt_version": "v2"}
    )["answer"]

    assert result.needs_human_review
    assert "DEMO-USE-01" in {item.clause_id for item in result.retrieved_passages}
    assert verify_assessment(result)
    assert {item["tool"] for item in result.tool_calls} >= {
        "corpus_search",
        "lookup_clause",
        "lookup_request_field",
    }


def test_unknown_loss_circumstances_escalate_without_claiming_coverage():
    fnol = scenario_fnol(
        "The vehicle was damaged by an unknown event not described in the policy corpus."
    )
    classification = classify(fnol).model_dump(mode="json")
    result = coverage_graph.invoke(
        {"fnol": fnol, "classification": classification, "prompt_version": "v2"}
    )["answer"]

    assert result.needs_human_review
    assert result.citations == []


def test_third_party_claim_is_classified_separately():
    fnol = scenario_fnol("My car hit another vehicle and damaged its bodywork.")
    assert classify(fnol).claim_type.value == "third_party"


def test_citation_must_resolve_to_retrieved_passage():
    assessment = CoverageAssessment(
        coverage_position="Advisory only",
        citations=[
            {
                "clause_id": "missing",
                "source": "not in corpus",
                "passage": "unsupported",
            }
        ],
        confidence=0.1,
        needs_human_review=False,
        retrieved_passages=[],
    )
    assert not verify_assessment(assessment)


def test_evaluation_set_has_at_least_twenty_scenarios():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    assert len(scenarios) >= 20
    assert any("food delivery" in item["description"].lower() for item in scenarios)
