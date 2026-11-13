import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, Request
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.models import Citation, CoverageAssessment
from app.telemetry import configure_tracing
from app.config import MODEL_STUB_DELAY_SECONDS
from app.logging_config import correlation_id
from opentelemetry import propagate

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus" / "demo_policy.json"
INDEX_PATH = Path("data/index")
COLLECTION = "motor-policy-clauses"


class HashingEmbeddings(Embeddings):
    """Small deterministic local embedder for the offline demo corpus."""

    dimension = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        words = re.findall(r"[a-z0-9]+", text.lower())
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def load_clauses() -> list[dict[str, str]]:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    clauses = list(data["clauses"])
    indexed_path = ROOT / "data" / "corpus" / "processed" / "indexed_corpus.json"
    if indexed_path.exists():
        clauses.extend(json.loads(indexed_path.read_text(encoding="utf-8"))["clauses"])
    return clauses


def build_store() -> Chroma:
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(INDEX_PATH),
        embedding_function=HashingEmbeddings(),
    )
    clauses = load_clauses()
    docs = [
        Document(
            page_content=clause["text"],
            metadata={
                "clause_id": clause["clause_id"],
                "source": clause["source"],
                "category": clause["category"],
                "topics": clause["topics"],
            },
        )
        for clause in clauses
    ]
    ids = [clause["clause_id"] for clause in clauses]
    existing = store.get()
    current_ids = existing["ids"]
    stale_ids = list(set(current_ids) - set(ids))
    if stale_ids:
        store.delete(ids=stale_ids)
    existing_documents = dict(zip(existing["ids"], existing["documents"] or []))
    changed_docs = [
        (document, clause["clause_id"])
        for document, clause in zip(docs, clauses)
        if existing_documents.get(clause["clause_id"]) != document.page_content
    ]
    if changed_docs:
        store.add_documents(
            [document for document, _ in changed_docs],
            ids=[clause_id for _, clause_id in changed_docs],
        )
    return store


store = build_store()


def _citation(document: Document) -> Citation:
    return Citation(
        clause_id=document.metadata["clause_id"],
        source=document.metadata["source"],
        passage=document.page_content,
        category=document.metadata["category"],
    )


@tool
def corpus_search(
    query: str, limit: int = 5, category: str | None = None
) -> list[dict[str, str]]:
    """Search the indexed policy and return clauses relevant to a claim question."""
    filters = {"category": category} if category else None
    documents = store.similarity_search(
        query, k=max(1, min(limit, 10)), filter=filters
    )
    return [_citation(document).model_dump() for document in documents]


@tool
def lookup_clause(clause_id: str) -> dict[str, str] | None:
    """Look up one exact clause by its stable clause identifier."""
    result = store.get(ids=[clause_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {
        "clause_id": result["ids"][0],
        "source": result["metadatas"][0]["source"],
        "passage": result["documents"][0],
        "category": result["metadatas"][0]["category"],
    }


@tool
def lookup_request_field(request_fields: dict[str, Any], field: str) -> str | None:
    """Read a field already supplied by the FNOL; do not infer missing values."""
    value = request_fields.get(field)
    return None if value is None else str(value)


TOOLS = [corpus_search, lookup_clause, lookup_request_field]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}


class CoverageInput(BaseModel):
    fnol: dict[str, Any]
    classification: dict[str, Any]
    prompt_version: str = "v2"


class CoverageState(TypedDict, total=False):
    fnol: dict[str, Any]
    classification: dict[str, Any]
    prompt_version: str
    retrieved: list[Citation]
    answer: CoverageAssessment
    search_count: int
    tool_calls: list[dict[str, str]]


def retrieve_applicable(state: CoverageState) -> CoverageState:
    claim_type = state["classification"]["claim_type"]
    query = f"{claim_type} accident damage cover policy"
    result = corpus_search.invoke(
        {
            "query": query,
            "limit": 4,
            "category": "coverage",
        }
    )
    state["tool_calls"] = [
        {
            "tool": "corpus_search",
            "query": query,
            "category": "coverage",
            "result_count": str(len(result)),
        }
    ]
    if result:
        exact = lookup_clause.invoke({"clause_id": result[0]["clause_id"]})
        state["tool_calls"].append(
            {"tool": "lookup_clause", "clause_id": result[0]["clause_id"]}
        )
        if exact:
            result[0] = exact
    state["retrieved"] = [Citation.model_validate(item) for item in result]
    state["search_count"] = 1
    return state


def check_modifiers(state: CoverageState) -> CoverageState:
    detected_use = lookup_request_field.invoke(
        {
            "request_fields": state["classification"]["entities"],
            "field": "vehicle_use_detected",
        }
    ) or "unknown"
    description = state["fnol"]["description"]
    usage_query = f"vehicle permitted use restriction exclusion {detected_use} {description}"
    modifier_docs = corpus_search.invoke(
        {
            "query": usage_query,
            "limit": 4,
            "category": "restriction",
        }
    )
    restriction_count = len(modifier_docs)
    exclusion_docs = corpus_search.invoke(
        {
            "query": usage_query,
            "limit": 4,
            "category": "exclusion",
        }
    )
    exclusion_count = len(exclusion_docs)
    modifier_docs.extend(exclusion_docs)
    unfiltered_docs = corpus_search.invoke(
        {"query": usage_query, "limit": 4}
    )
    modifier_docs.extend(unfiltered_docs)
    state.setdefault("tool_calls", []).extend(
        [
            {
                "tool": "lookup_request_field",
                "field": "vehicle_use_detected",
                "value": detected_use,
            },
            {
                "tool": "corpus_search",
                "query": usage_query,
                "category": "restriction",
                "result_count": str(restriction_count),
            },
            {
                "tool": "corpus_search",
                "query": usage_query,
                "category": "exclusion",
                "result_count": str(exclusion_count),
            },
            {
                "tool": "corpus_search",
                "query": usage_query,
                "category": "all",
                "result_count": str(len(unfiltered_docs)),
            },
        ]
    )
    if modifier_docs:
        exact = lookup_clause.invoke({"clause_id": modifier_docs[0]["clause_id"]})
        state["tool_calls"].append(
            {"tool": "lookup_clause", "clause_id": modifier_docs[0]["clause_id"]}
        )
        if exact:
            modifier_docs[0] = exact
    by_id = {citation.clause_id: citation for citation in state.get("retrieved", [])}
    for item in modifier_docs:
        citation = Citation.model_validate(item)
        by_id[citation.clause_id] = citation
    state["retrieved"] = list(by_id.values())
    state["search_count"] = 2
    return state


def assess(state: CoverageState) -> CoverageState:
    retrieved = state.get("retrieved", [])
    claim_type = state["classification"]["claim_type"]
    detected_use = state["classification"]["entities"]["vehicle_use_detected"]
    demo_coverage = [
        item for item in retrieved
        if item.clause_id.startswith("DEMO-COVER-")
    ]
    demo_clause_by_type = {
        "own_damage": "DEMO-COVER-OD-01",
        "total_loss": "DEMO-COVER-OD-01",
        "third_party": "DEMO-COVER-TP-01",
        "theft": "DEMO-COVER-TH-01",
    }
    preferred_clause = demo_clause_by_type.get(claim_type)
    if preferred_clause:
        coverage = [item for item in demo_coverage if item.clause_id == preferred_clause]
    else:
        coverage = [
            item for item in retrieved
            if item.category == "coverage"
        ][:1]
    restrictions = [
        item for item in retrieved
        if item.category in {"restriction", "exclusion"}
    ]
    delivery_use = detected_use in {
        "food_delivery",
        "commercial",
        "taxi",
        "restricted_activity",
    }
    unsupported_circumstances = (
        state["classification"]["entities"].get("possible_unsupported_circumstances")
        == "true"
    )
    weak = not coverage or claim_type == "unknown" or unsupported_circumstances
    review = weak or delivery_use
    if weak:
        position = "The available demo corpus does not support a reliable coverage position."
        reason = "The claim or applicable cover is outside the available evidence."
        citations = []
    elif delivery_use:
        position = "Reported vehicle use may be restricted; a handler must review the policy and facts."
        reason = "Reported vehicle use or activity may be restricted by the policy wording."
        citations = coverage[:1] + restrictions[:1]
    else:
        position = "The retrieved demo wording appears relevant; a handler must confirm applicability."
        reason = None
        citations = coverage[:1]
    state["answer"] = CoverageAssessment(
        coverage_position=position,
        citations=citations,
        confidence=0.3 if weak else 0.72,
        needs_human_review=review,
        reason=reason,
        retrieved_passages=retrieved,
        tool_calls=state.get("tool_calls", []),
    )
    return state


def build_graph():
    graph = StateGraph(CoverageState)
    graph.add_node("retrieve_applicable", retrieve_applicable)
    graph.add_node("check_modifiers", check_modifiers)
    graph.add_node("assess", assess)
    graph.add_edge(START, "retrieve_applicable")
    graph.add_conditional_edges(
        "retrieve_applicable",
        lambda state: (
            "check_modifiers"
            if state.get("prompt_version") == "v2" and state.get("retrieved")
            else "assess"
        ),
        {"check_modifiers": "check_modifiers", "assess": "assess"},
    )
    graph.add_edge("check_modifiers", "assess")
    graph.add_edge("assess", END)
    return graph.compile()


coverage_graph = build_graph()
app = FastAPI(title="Policy Coverage Agent", version="1.0.0")
tracer = configure_tracing("coverage-agent")


@app.middleware("http")
async def trace_request(request: Request, call_next):
    token = correlation_id.set(request.headers.get("X-Correlation-ID", "-"))
    parent_context = propagate.extract(dict(request.headers))
    try:
        with tracer.start_as_current_span("http.request", context=parent_context):
            return await call_next(request)
    finally:
        correlation_id.reset(token)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "coverage", "indexed_clauses": str(len(load_clauses()))}


@app.post("/assess", response_model=CoverageAssessment)
async def assess_request(payload: CoverageInput) -> CoverageAssessment:
    with tracer.start_as_current_span("coverage.assess"):
        await asyncio.sleep(MODEL_STUB_DELAY_SECONDS)
        result = coverage_graph.invoke(
            {
                "fnol": payload.fnol,
                "classification": payload.classification,
                "prompt_version": payload.prompt_version,
            }
        )
    return result["answer"]
