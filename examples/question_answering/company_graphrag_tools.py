"""Tool-routed GraphRAG over the GNEM Company graph with local Ollama models.

Text2Cypher is attempted first by default so exact graph fields such as tier,
employment, county, roles, OEMs and products are returned as explicit context.
If Text2Cypher returns no records, the generation LLM (qwen3:14b) can still route
to another retriever via tool calling (``ToolsRetriever`` ->
``OllamaLLM.invoke_with_tools``):

- ``text2cypher_search``   : exact/structured questions (tiers, employment thresholds,
                             counts, "list all/every", specific counties, sole-source).
- ``hybrid_search``        : keyword + semantic (company names, product terms such as
                             "copper foil", "DC-to-DC converter").
- ``vector_cypher_search`` : semantic match + graph expansion (roles, OEMs, locations).
- ``vector_search``        : plain semantic similarity over company profiles.

All retrieval tools use ``top_k = 50``. If the router selects no tool, a hybrid
fallback runs so the answer is always grounded in retrieved context.

This module is imported by ``examples/apps/company_chat_app.py`` and can be run
directly as a smoke test::

    uv run --extra ollama python examples/question_answering/company_graphrag_tools.py
    uv run --extra ollama python examples/question_answering/company_graphrag_tools.py \
        --question 'Which county has the highest total employment across all companies?'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import time
import traceback
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import neo4j
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.retrievers import (
    HybridRetriever,
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.tool import ObjectParameter, StringParameter, Tool
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem


DEFAULT_URI = "neo4j+s://945619b7.databases.neo4j.io"
DEFAULT_USERNAME = "945619b7"
DEFAULT_DATABASE = "945619b7"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
DEFAULT_GENERATION_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_VECTOR_INDEX = "company_embeddings"
DEFAULT_FULLTEXT_INDEX = "company_fulltext"
DEFAULT_TOP_K = 50

# Schema surfaced to the Text2Cypher tool. Kept small and accurate so the model
# writes valid Cypher against the GNEM graph.
COMPANY_SCHEMA = """
Node properties:
Company {company: STRING, product_service_text: STRING, employment_total: INTEGER,
         employment: INTEGER, category: STRING}
Category {category: STRING}                         // supplier tier e.g. "Tier 1", "Tier 1/2", "Tier 2/3"
SupplyChainRole {ev_supply_chain_role: STRING}
IndustryGroup {industry_group: STRING}
Location {location: STRING}                          // "City, County" e.g. "Gainesville, Hall County". Every company is in the U.S. state of Georgia; this value NEVER contains the word "Georgia".
OEMGroup {primary_oems: STRING}
FacilityType {primary_facility_type: STRING}
EVRelevance {ev_battery_relevant: STRING}            // "Direct", "Indirect", "No"

Relationships:
(:Company)-[:HAS_CATEGORY]->(:Category)
(:Company)-[:HAS_SUPPLY_CHAIN_ROLE]->(:SupplyChainRole)
(:Company)-[:BELONGS_TO_INDUSTRY]->(:IndustryGroup)
(:Company)-[:LOCATED_IN]->(:Location)
(:Company)-[:SUPPLIES_TO]->(:OEMGroup)
(:Company)-[:HAS_FACILITY_TYPE]->(:FacilityType)
(:Company)-[:HAS_EV_RELEVANCE]->(:EVRelevance)
"""

TEXT2CYPHER_EXAMPLES = [
    (
        "USER INPUT: 'List five companies and their locations' "
        "QUERY: MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
        "RETURN c.company AS company, l.location AS location LIMIT 5"
    ),
    (
        "USER INPUT: 'Which companies supply to Hyundai?' "
        "QUERY: MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEMGroup) "
        "WHERE toLower(o.primary_oems) CONTAINS 'hyundai' "
        "RETURN c.company AS company, o.primary_oems AS oem LIMIT 50"
    ),
    (
        "USER INPUT: 'Show all Tier 1/2 suppliers with their role and product' "
        "QUERY: MATCH (c:Company)-[:HAS_CATEGORY]->(cat:Category) "
        "WHERE cat.category = 'Tier 1/2' "
        "OPTIONAL MATCH (c)-[:HAS_SUPPLY_CHAIN_ROLE]->(r:SupplyChainRole) "
        "RETURN c.company AS company, cat.category AS tier, "
        "collect(DISTINCT r.ev_supply_chain_role) AS roles, "
        "c.product_service_text AS product LIMIT 50"
    ),
    (
        "USER INPUT: 'Which companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?' "
        "QUERY: MATCH (c:Company)-[:HAS_SUPPLY_CHAIN_ROLE]->(r:SupplyChainRole) "
        "WHERE toLower(r.ev_supply_chain_role) CONTAINS 'battery cell' "
        "OR toLower(r.ev_supply_chain_role) CONTAINS 'battery pack' "
        "OPTIONAL MATCH (c)-[:HAS_CATEGORY]->(cat:Category) "
        "RETURN c.company AS company, "
        "collect(DISTINCT r.ev_supply_chain_role) AS roles, "
        "collect(DISTINCT cat.category) AS tiers LIMIT 50"
    ),
    (
        "USER INPUT: 'Which county has the highest total employment across all companies?' "
        "QUERY: MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
        "WITH split(l.location, ', ')[-1] AS county, sum(c.employment_total) AS total "
        "RETURN county, total ORDER BY total DESC LIMIT 1"
    ),
]

TEXT2CYPHER_PROMPT = """
You generate read-only Cypher for Neo4j.
Return only one Cypher statement. Do not include markdown, backticks, comments,
explanations, XML tags, or chain-of-thought.
Use only the schema below.
Every company is already in the U.S. state of Georgia. The Location value is
"City, County" and never contains the word "Georgia", so NEVER filter locations
by 'georgia' or by state. A county filter uses the county name, e.g.
toLower(l.location) CONTAINS 'gwinnett'.
When the result is a list of companies, ALWAYS return c.company AS company and the
filtering field(s) so the answer can show them - e.g. cat.category AS tier when the
question is about tiers, or the role/OEM/industry being filtered on. (Do not add
these to aggregate queries such as counts or sums.)
If the WHERE clause filters SupplyChainRole, the RETURN clause MUST include
collect(DISTINCT r.ev_supply_chain_role) AS roles. If the question asks for tier,
also return cat.category AS tier or collect(DISTINCT cat.category) AS tiers.
Prefer LIMIT 50 unless the user asks for a different number or an aggregate.

Schema:
{schema}

Examples:
{examples}

Question:
{query_text}

Cypher:
"""

# Vector + graph expansion: semantic hit on Company, then pull neighbouring context.
VECTOR_CYPHER_QUERY = """
WITH node, score
OPTIONAL MATCH (node)-[:HAS_SUPPLY_CHAIN_ROLE]-(r:SupplyChainRole)
OPTIONAL MATCH (node)-[:SUPPLIES_TO]-(o:OEMGroup)
OPTIONAL MATCH (node)-[:LOCATED_IN]-(l:Location)
OPTIONAL MATCH (node)-[:BELONGS_TO_INDUSTRY]-(i:IndustryGroup)
OPTIONAL MATCH (node)-[:HAS_CATEGORY]-(cat:Category)
RETURN
  node.company AS company,
  node.product_service_text AS product_service,
  node.employment_total AS employment,
  collect(DISTINCT r.ev_supply_chain_role) AS ev_supply_chain_roles,
  collect(DISTINCT o.primary_oems) AS primary_oems,
  collect(DISTINCT l.location) AS locations,
  collect(DISTINCT i.industry_group) AS industry_groups,
  collect(DISTINCT cat.category) AS categories,
  score
"""

# Deterministic pre-router: questions with these signals are structured/aggregation
# queries that a small model tends to mis-route. Text2Cypher is also used as the
# default first attempt for all questions in RoutingRetriever.
_STRUCTURED_HINT = re.compile(
    r"\b(all|every|list|how many|count|number of|highest|largest|lowest|smallest"
    r"|most|least|top|fewer than|more than|at least|over|under|below|above"
    r"|employment|employees|workers|tier\s*[123]|county|counties"
    r"|sole[- ]?source|single (company|point|supplier)|only a single)\b",
    re.IGNORECASE,
)


def looks_structured(query_text: str) -> bool:
    """Heuristic: does this question need exact filtering/aggregation (text2cypher)?"""
    return bool(_STRUCTURED_HINT.search(query_text or ""))


# Faithful answer prompt: answer from the retrieved records, include the companies
# that match the question (the tier / filter field is now present in the context so
# the model can judge each record), and do not invent companies. Deliberately does
# NOT tell the model to blindly include every record - the semantic retrievers return
# near-matches that may not actually satisfy the question.
ANSWER_SYSTEM_INSTRUCTION = (
    "You answer questions about Georgia EV supply-chain companies using ONLY the "
    "retrieved records provided. List the companies that match the question, with the "
    "details it asks for, and do not include companies that do not match or that are "
    "not in the records. Do not omit or summarise matching companies. If no record "
    "matches, say so."
)

ANSWER_TEMPLATE = """Context (retrieved records):
{context}

Examples:
{examples}

Question:
{query_text}

Answer:
"""


def build_rag_template() -> RagTemplate:
    return RagTemplate(
        template=ANSWER_TEMPLATE,
        system_instructions=ANSWER_SYSTEM_INSTRUCTION,
    )


ROUTER_SYSTEM_INSTRUCTION = (
    "You route questions about Georgia EV supply-chain companies to exactly one "
    "retrieval tool, then stop. Choose:\n"
    "- text2cypher_search for exact/structured questions: filtering by tier or "
    "category, employment thresholds, counts, 'list all/every', specific counties, "
    "sole-source / single-supplier, or anything needing precise aggregation.\n"
    "- hybrid_search for specific company names or product/technology terms "
    "(e.g. 'copper foil', 'DC-to-DC converter', 'powder coating').\n"
    "- vector_cypher_search for open-ended semantic questions that also need related "
    "graph context (roles, OEMs, locations) around similar companies.\n"
    "- vector_search for broad 'find companies like...' semantic questions.\n"
    "Always call one tool with the user's question as query_text."
)


@dataclass
class Settings:
    uri: str
    username: str
    password: str
    database: Optional[str]
    embedding_model: str
    generation_model: str
    ollama_host: str
    vector_index_name: str
    fulltext_index_name: str


class NoThinkOllamaLLM(OllamaLLM):
    """OllamaLLM that strips Qwen ``<think>...</think>`` blocks before parsing.

    Used for Text2Cypher so the generated Cypher is not polluted by reasoning text.
    """

    def invoke(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = super().invoke(*args, **kwargs)
        response.content = _strip_thinking(response.content)
        return response


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        uri=os.getenv("NEO4J_URI", DEFAULT_URI),
        username=os.getenv("NEO4J_USERNAME", DEFAULT_USERNAME),
        password=get_required_env("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE) or None,
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        generation_model=os.getenv("OLLAMA_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        vector_index_name=os.getenv("VECTOR_INDEX_NAME", DEFAULT_VECTOR_INDEX),
        fulltext_index_name=os.getenv("FULLTEXT_INDEX_NAME", DEFAULT_FULLTEXT_INDEX),
    )


def build_embedder(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, host=settings.ollama_host)


# Large enough context so all top_k=50 retrieved records fit in the prompt, and
# enough output budget to list every matching company (a small num_ctx silently
# truncates the context and yields partial "show all" answers).
_LLM_OPTIONS = {"temperature": 0, "num_ctx": 12000, "num_predict": 2048}

# Keep parameters compatible with the installed Ollama Python client. Some newer
# clients support a top-level think=False flag, but this environment's client does
# not; NoThinkOllamaLLM strips Qwen <think> blocks instead.
def _model_params() -> dict[str, Any]:
    return {"options": dict(_LLM_OPTIONS)}


def build_llm(settings: Settings) -> OllamaLLM:
    return NoThinkOllamaLLM(
        model_name=settings.generation_model,
        model_params=_model_params(),
        host=settings.ollama_host,
    )


def build_text2cypher_llm(settings: Settings) -> OllamaLLM:
    return NoThinkOllamaLLM(
        model_name=settings.generation_model,
        model_params=_model_params(),
        host=settings.ollama_host,
    )


def build_retrievers(
    driver: neo4j.Driver, settings: Settings, embedder: OllamaEmbeddings
) -> dict[str, Any]:
    vector = VectorRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=embedder,
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )
    hybrid = HybridRetriever(
        driver=driver,
        vector_index_name=settings.vector_index_name,
        fulltext_index_name=settings.fulltext_index_name,
        embedder=embedder,
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )
    vector_cypher = VectorCypherRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        retrieval_query=VECTOR_CYPHER_QUERY,
        embedder=embedder,
        neo4j_database=settings.database,
    )

    def text2cypher_formatter(record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=json.dumps(record.data(), ensure_ascii=False, sort_keys=True),
            metadata={},
        )

    text2cypher = Text2CypherRetriever(
        driver=driver,
        llm=build_text2cypher_llm(settings),
        neo4j_schema=COMPANY_SCHEMA,
        examples=TEXT2CYPHER_EXAMPLES,
        result_formatter=text2cypher_formatter,
        custom_prompt=TEXT2CYPHER_PROMPT,
        neo4j_database=settings.database,
    )
    return {
        "vector": vector,
        "hybrid": hybrid,
        "vector_cypher": vector_cypher,
        "text2cypher": text2cypher,
    }


# Lucene reserved characters that break db.index.fulltext.queryNodes when a raw
# question is used as the query (e.g. '"Tier 1/2" ... Product / Service.').
_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/&|]')


def sanitize_fulltext(text: str) -> str:
    """Turn a raw question into a safe full-text query (plain keywords)."""
    cleaned = _LUCENE_SPECIAL.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _query_tool(
    name: str, description: str, search_fn: Callable[[str], RetrieverResult]
) -> Tool:
    """Wrap a retriever search as a Tool exposing only ``query_text`` (top_k fixed)."""

    def execute(query_text: str, **_: Any) -> RetrieverResult:
        return search_fn(query_text)

    return Tool(
        name=name,
        description=description,
        execute_func=execute,
        parameters=ObjectParameter(
            description=f"{name} parameters",
            properties={
                "query_text": StringParameter(
                    description="The user's natural-language question, passed verbatim.",
                    required=True,
                )
            },
            required_properties=["query_text"],
            additional_properties=False,
        ),
    )


def build_tools(retrievers: dict[str, Any], top_k: int = DEFAULT_TOP_K) -> list[Tool]:
    return [
        _query_tool(
            "text2cypher_search",
            "Exact, structured questions: filter by supplier tier/category, employment "
            "thresholds, counts, 'list all/every', specific counties, sole-source or "
            "single-supplier analysis. Generates and runs a Cypher query.",
            lambda q: retrievers["text2cypher"].search(query_text=q),
        ),
        _query_tool(
            "hybrid_search",
            "Specific company names or product/technology terms (e.g. 'copper foil', "
            "'DC-to-DC converter', 'powder coating'). Combines full-text and vector search.",
            lambda q: retrievers["hybrid"].search(
                query_text=sanitize_fulltext(q), top_k=top_k
            ),
        ),
        _query_tool(
            "vector_cypher_search",
            "Open-ended semantic questions that also need related graph context "
            "(supply-chain roles, OEMs, locations) around similar companies.",
            lambda q: retrievers["vector_cypher"].search(query_text=q, top_k=top_k),
        ),
        _query_tool(
            "vector_search",
            "Broad 'find companies like...' semantic similarity over company profiles.",
            lambda q: retrievers["vector"].search(query_text=q, top_k=top_k),
        ),
    ]


class RoutingRetriever(ToolsRetriever):
    """ToolsRetriever with Text2Cypher default, clean formatter and fallback.

    Routing order per question:
    1. Default: if ``structured_search`` is configured, run Text2Cypher first.
       This makes exact graph fields such as tier visible in retrieved context.
    2. LLM tool calling: if Text2Cypher found nothing, let the LLM pick a tool via
       ``invoke_with_tools``.
    3. Fallback: if still no context, run a hybrid search so the answer is grounded.

    Records are formatted down to their retrieved text (``content``) so GraphRAG
    builds a readable context string.
    """

    def __init__(
        self,
        *args: Any,
        structured_search: Optional[Callable[[str], RetrieverResult]] = None,
        fallback_search: Optional[Callable[[str], RetrieverResult]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._structured_search = structured_search
        self._fallback_search = fallback_search
        # Picked up by Retriever.get_result_formatter().
        self.result_formatter = self._format_record

    @staticmethod
    def _format_record(record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=record.get("content"),
            metadata=record.get("metadata"),
        )

    @staticmethod
    def _extract_filter_clause(cypher: str) -> str:
        compact = re.sub(r"\s+", " ", cypher).strip()
        clauses = re.findall(
            r"\bWHERE\b\s+(.*?)(?=\bOPTIONAL MATCH\b|\bMATCH\b|\bWITH\b|\bRETURN\b|\bORDER BY\b|\bLIMIT\b|$)",
            compact,
            flags=re.IGNORECASE,
        )
        return " AND ".join(clause.strip() for clause in clauses if clause.strip())

    @staticmethod
    def _records_from_result(
        result: RetrieverResult, tool_name: str
    ) -> list[neo4j.Record]:
        records = []
        cypher = (result.metadata or {}).get("cypher")
        if tool_name == "text2cypher_search" and cypher:
            filter_clause = RoutingRetriever._extract_filter_clause(cypher)
            evidence = (
                "RETRIEVAL_EVIDENCE "
                f"tool={tool_name}; "
                f"filter_used={filter_clause or '<none>'}; "
                f"generated_cypher={cypher}"
            )
            records.append(
                neo4j.Record(
                    {
                        "content": evidence,
                        "tool_name": tool_name,
                        "metadata": {
                            "tool": tool_name,
                            "record_type": "retrieval_evidence",
                            "cypher": cypher,
                            "filter_used": filter_clause,
                        },
                    }
                )
            )
        records.extend(
            neo4j.Record(
                {
                    "content": item.content,
                    "tool_name": tool_name,
                    "metadata": {**(item.metadata or {}), "tool": tool_name},
                }
            )
            for item in result.items
        )
        return records

    def get_search_results(
        self,
        query_text: str,
        message_history: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> RawSearchResult:
        # 1. Text2Cypher is the default first attempt for every question.
        if self._structured_search is not None:
            try:
                structured = self._structured_search(query_text)
            except Exception:
                structured = None  # fall through to LLM routing on any error
            if structured is not None and structured.items:
                return RawSearchResult(
                    records=self._records_from_result(structured, "text2cypher_search"),
                    metadata={
                        "query": query_text,
                        "router": (
                            "default_text2cypher"
                            if not looks_structured(query_text)
                            else "pre_router_heuristic"
                        ),
                        "tools_selected": ["text2cypher_search"],
                    },
                )

        # 2. LLM tool calling.
        result = super().get_search_results(query_text, message_history, **kwargs)
        if result.records or self._fallback_search is None:
            return result

        # 3. Grounded fallback.
        try:
            fallback = self._fallback_search(query_text)
        except Exception as exc:  # never let a fallback failure crash the search
            metadata = dict(result.metadata or {})
            metadata["fallback_error"] = f"{type(exc).__name__}: {exc}"
            return RawSearchResult(records=[], metadata=metadata)

        metadata = dict(result.metadata or {})
        metadata["fallback"] = "hybrid_search"
        metadata["tools_selected"] = ["fallback_hybrid"]
        return RawSearchResult(
            records=self._records_from_result(fallback, "fallback_hybrid"),
            metadata=metadata,
        )


def build_engine(
    driver: neo4j.Driver, settings: Settings, top_k: int = DEFAULT_TOP_K
) -> tuple[GraphRAG, RoutingRetriever]:
    """Build the tool-routed GraphRAG engine. Returns (rag, router)."""
    embedder = build_embedder(settings)
    retrievers = build_retrievers(driver, settings, embedder)
    tools = build_tools(retrievers, top_k=top_k)
    llm = build_llm(settings)
    router = RoutingRetriever(
        driver=driver,
        llm=llm,
        tools=tools,
        neo4j_database=settings.database,
        system_instruction=ROUTER_SYSTEM_INSTRUCTION,
        structured_search=lambda q: retrievers["text2cypher"].search(query_text=q),
        fallback_search=lambda q: retrievers["hybrid"].search(
            query_text=sanitize_fulltext(q), top_k=top_k
        ),
    )
    rag = GraphRAG(retriever=router, llm=llm, prompt_template=build_rag_template())
    return rag, router


def build_text2cypher_engine(driver: neo4j.Driver, settings: Settings) -> GraphRAG:
    """Build a GraphRAG engine that uses ONLY Text2Cypher (no router, no fallback).

    Every question is answered by generating and running Cypher against the graph;
    if the generated query returns no rows, the answer is grounded in an empty
    context (no semantic fallback). Use this to evaluate the Text2Cypher path alone.
    """
    text2cypher = build_retrievers(driver, settings, build_embedder(settings))[
        "text2cypher"
    ]
    llm = build_llm(settings)
    return GraphRAG(
        retriever=text2cypher, llm=llm, prompt_template=build_rag_template()
    )


SAMPLE_QUESTIONS = [
    'Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service.',
    "Find Georgia-based companies that manufacture copper foil or electrodeposited "
    "materials suitable for EV battery current collectors.",
    "Which county has the highest total employment across all companies, and what is "
    "the combined employment in that county?",
]


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_questions(path: pathlib.Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip().lower() != "question"
    ]


def make_output_dir(output_dir: Optional[pathlib.Path]) -> pathlib.Path:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = pathlib.Path("runs") / f"company_qa_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: pathlib.Path, data: Any) -> None:
    path.write_text(
        json.dumps(make_jsonable(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")
        file.flush()


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "question",
        "status",
        "mode",
        "tool_used",
        "retrieved_item_count",
        "answer",
        "cypher",
        "duration_seconds",
        "error_type",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row.get("retriever_metadata") or {}
            error = row.get("error") or {}
            writer.writerow(
                {
                    "question_id": row.get("question_id"),
                    "question": row.get("question"),
                    "status": row.get("status"),
                    "mode": row.get("mode"),
                    "tool_used": row.get("tool_used"),
                    "retrieved_item_count": row.get("retrieved_item_count"),
                    "answer": row.get("answer"),
                    "cypher": metadata.get("cypher", ""),
                    "duration_seconds": row.get("duration_seconds"),
                    "error_type": error.get("type", ""),
                    "error_message": error.get("message", ""),
                }
            )


def write_eval_exports(output_dir: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    ragas_path = output_dir / "ragas_dataset.jsonl"
    ragas_path.write_text("", encoding="utf-8")
    deepeval_cases = []
    for row in rows:
        contexts = [item["content"] for item in row.get("retrieved_context", [])]
        metadata = {
            "question_id": row.get("question_id"),
            "mode": row.get("mode"),
            "tool_used": row.get("tool_used"),
            "status": row.get("status"),
            "retriever_metadata": row.get("retriever_metadata"),
        }
        append_jsonl(
            ragas_path,
            {
                "question": row.get("question"),
                "answer": row.get("answer") or "",
                "contexts": contexts,
                "ground_truth": None,
                "metadata": metadata,
            },
        )
        deepeval_cases.append(
            {
                "input": row.get("question"),
                "actual_output": row.get("answer") or "",
                "retrieval_context": contexts,
                "context": contexts,
                "metadata": metadata,
            }
        )
    write_json(output_dir / "deepeval_dataset.json", deepeval_cases)


def run_batch(
    rag: GraphRAG,
    questions: list[tuple[int, str]],
    *,
    output_dir: pathlib.Path,
    top_k: int,
    settings: Settings,
    mode: str = "tool-routed-text2cypher-default",
) -> None:
    answers_jsonl = output_dir / "answers.jsonl"
    answers_jsonl.write_text("", encoding="utf-8")
    (output_dir / "questions.txt").write_text(
        "\n".join(question for _, question in questions) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "mode": mode,
            "top_k": top_k,
            "settings": asdict(settings) | {"password": "<omitted>"},
            "outputs": {
                "answers_jsonl": str(answers_jsonl),
                "answers_csv": str(output_dir / "answers.csv"),
                "ragas_dataset_jsonl": str(output_dir / "ragas_dataset.jsonl"),
                "deepeval_dataset_json": str(output_dir / "deepeval_dataset.json"),
            },
        },
    )

    rows = []
    for question_id, question in questions:
        started_at = datetime.now(timezone.utc)
        start = time.perf_counter()
        row: dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "mode": mode,
            "status": "ok",
            "started_at": started_at.isoformat(),
            "retriever_config": {"top_k": top_k},
        }
        try:
            result = rag.search(question, retriever_config={}, return_context=True)
            retriever_result = result.retriever_result
            metadata = retriever_result.metadata or {}
            tools_selected = metadata.get("tools_selected") or []
            context_items = [
                {
                    "rank": index,
                    "content": item.content,
                    "metadata": item.metadata or {},
                }
                for index, item in enumerate(retriever_result.items, start=1)
            ]
            row.update(
                {
                    "tool_used": ", ".join(tools_selected)
                    or metadata.get("__retriever")
                    or "<none>",
                    "retriever_metadata": metadata,
                    "retrieved_context": context_items,
                    "retrieved_item_count": len(context_items),
                    "answer": result.answer.strip(),
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["answer"] = ""
            row["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            row["duration_seconds"] = round(time.perf_counter() - start, 3)

        rows.append(row)
        append_jsonl(answers_jsonl, row)
        print(
            f"[{question_id}] {row['status']} - "
            f"{row.get('retrieved_item_count', 0)} context items - "
            f"{row['duration_seconds']}s"
        )

    write_json(output_dir / "answers.json", rows)
    write_csv(output_dir / "answers.csv", rows)
    write_eval_exports(output_dir, rows)
    passed = sum(1 for row in rows if row["status"] == "ok")
    print(f"\nDone: {passed}/{len(rows)} questions completed.")
    print(f"Full JSONL: {answers_jsonl}")
    print(f"CSV summary: {output_dir / 'answers.csv'}")
    print(f"RAGAS JSONL: {output_dir / 'ragas_dataset.jsonl'}")
    print(f"DeepEval JSON: {output_dir / 'deepeval_dataset.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-routed GraphRAG smoke test.")
    parser.add_argument("--question", default=None, help="Single question to run.")
    parser.add_argument("--questions", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--text2cypher-only",
        action="store_true",
        help="Use ONLY Text2Cypher (no router, no fallback) for every question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    driver = GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password))
    try:
        driver.verify_connectivity()
        if args.text2cypher_only:
            rag = build_text2cypher_engine(driver, settings)
            run_mode = "text2cypher-only"
        else:
            rag, _ = build_engine(driver, settings, top_k=args.top_k)
            run_mode = "tool-routed-text2cypher-default"
        if args.questions is not None:
            loaded_questions = list(enumerate(load_questions(args.questions), start=1))
            selected_questions = [
                (question_id, question)
                for question_id, question in loaded_questions
                if question_id >= args.start_at
            ]
            if args.limit is not None:
                selected_questions = selected_questions[: args.limit]
            run_batch(
                rag,
                selected_questions,
                output_dir=make_output_dir(args.output_dir),
                top_k=args.top_k,
                settings=settings,
                mode=run_mode,
            )
            return

        questions = [args.question] if args.question else SAMPLE_QUESTIONS
        for question in questions:
            result = rag.search(question, retriever_config={}, return_context=True)
            metadata = result.retriever_result.metadata or {}
            tools_selected = metadata.get("tools_selected", [])
            print("\n" + "=" * 80)
            print("Q:", question)
            print("Routed to:", tools_selected or "<none>")
            print("Retrieved items:", len(result.retriever_result.items))
            print("Answer:\n", result.answer.strip() or "<empty>")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
