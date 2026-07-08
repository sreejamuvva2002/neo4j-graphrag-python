"""Smoke-test Company GraphRAG retrieval modes with local Ollama models.

Required environment:
    NEO4J_PASSWORD

Optional environment overrides:
    NEO4J_URI
    NEO4J_USERNAME
    NEO4J_DATABASE
    OLLAMA_EMBEDDING_MODEL
    OLLAMA_GENERATION_MODEL
    OLLAMA_HOST
    VECTOR_INDEX_NAME
    FULLTEXT_INDEX_NAME

Examples:
    uv run --extra ollama python examples/question_answering/test_company_retrieval_modes.py --mode all
    uv run --extra ollama python examples/question_answering/test_company_retrieval_modes.py --mode vector --question "Which companies supply OEMs?"
"""

from __future__ import annotations

import argparse
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import neo4j
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.retrievers import (
    HybridRetriever,
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.tool import IntegerParameter, ObjectParameter, StringParameter, Tool


DEFAULT_URI = "neo4j+s://945619b7.databases.neo4j.io"
DEFAULT_USERNAME = "945619b7"
DEFAULT_DATABASE = "945619b7"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
DEFAULT_GENERATION_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_VECTOR_INDEX = "company_embeddings"
DEFAULT_FULLTEXT_INDEX = "company_fulltext"
DEFAULT_QUESTION = "Which companies supply OEMs in the EV supply chain?"

COMPANY_SCHEMA = """
Node properties:
Company {company: STRING, embedding_text: STRING}
SupplyChainRole {ev_supply_chain_role: STRING}
IndustryGroup {industry_group: STRING}
Location {location: STRING}
OEMGroup {primary_oems: STRING}
Category {category: STRING}

Relationships:
(:Company)-[:HAS_SUPPLY_CHAIN_ROLE]-(:SupplyChainRole)
(:Company)-[:BELONGS_TO_INDUSTRY]-(:IndustryGroup)
(:Company)-[:LOCATED_IN]-(:Location)
(:Company)-[:SUPPLIES_TO]-(:OEMGroup)
(:Company)-[:HAS_CATEGORY]-(:Category)
"""

TEXT2CYPHER_EXAMPLES = [
    (
        "USER INPUT: 'List five companies and their locations' "
        "QUERY: MATCH (c:Company)-[:LOCATED_IN]-(l:Location) "
        "RETURN c.company AS company, l.location AS location LIMIT 5"
    ),
    (
        "USER INPUT: 'Which companies supply to Volvo?' "
        "QUERY: MATCH (c:Company)-[:SUPPLIES_TO]-(o:OEMGroup) "
        "WHERE toLower(o.primary_oems) CONTAINS 'volvo' "
        "RETURN c.company AS company, o.primary_oems AS oem LIMIT 10"
    ),
]

TEXT2CYPHER_PROMPT = """
You generate read-only Cypher for Neo4j.
Return only one Cypher statement. Do not include markdown, backticks, comments,
explanations, XML tags, or chain-of-thought.
Use only the schema below.
Always include LIMIT 10 unless the user asks for a different limit.

Schema:
{schema}

Examples:
{examples}

Question:
{query_text}

Cypher:
"""


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
    """Tiny wrapper to remove Qwen thinking blocks before Text2Cypher parsing."""

    def invoke(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = super().invoke(*args, **kwargs)
        response.content = strip_thinking(response.content)
        return response


def strip_thinking(text: str) -> str:
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


def build_llm(settings: Settings) -> OllamaLLM:
    return OllamaLLM(
        model_name=settings.generation_model,
        model_params={"options": {"temperature": 0}},
        host=settings.ollama_host,
    )


def build_text2cypher_llm(settings: Settings) -> OllamaLLM:
    return NoThinkOllamaLLM(
        model_name=settings.generation_model,
        model_params={"options": {"temperature": 0}},
        host=settings.ollama_host,
    )


def build_embedder(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, host=settings.ollama_host)


def compact_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not metadata:
        return {}
    compact = dict(metadata)
    if "query_vector" in compact:
        compact["query_vector"] = f"{len(compact['query_vector'])} dimensions"
    return compact


def print_answer(
    title: str,
    answer: str,
    metadata: Optional[dict[str, Any]],
    item_count: Optional[int] = None,
) -> None:
    print(f"\n[{title}] PASS")
    if item_count is not None:
        print(f"Retrieved items: {item_count}")
    print("Answer:")
    print(answer.strip() or "<empty answer>")
    if metadata:
        print("Metadata:")
        print(compact_metadata(metadata))


def assert_retrieved(title: str, result: Any) -> int:
    if result.retriever_result is None:
        raise RuntimeError(f"{title} did not return retriever context")
    item_count = len(result.retriever_result.items)
    if item_count == 0:
        raise RuntimeError(f"{title} retrieved no context items")
    return item_count


def run_vector(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    retriever = VectorRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=build_embedder(settings),
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    result = rag.search(
        question, retriever_config={"top_k": top_k}, return_context=True
    )
    item_count = assert_retrieved("vector", result)
    print_answer("vector", result.answer, result.retriever_result.metadata, item_count)


def run_hybrid(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name=settings.vector_index_name,
        fulltext_index_name=settings.fulltext_index_name,
        embedder=build_embedder(settings),
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    result = rag.search(
        question, retriever_config={"top_k": top_k}, return_context=True
    )
    item_count = assert_retrieved("hybrid", result)
    print_answer("hybrid", result.answer, result.retriever_result.metadata, item_count)


def run_vector_cypher(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    retrieval_query = """
    WITH node, score
    OPTIONAL MATCH (node)-[:SUPPLIES_TO]-(o:OEMGroup)
    OPTIONAL MATCH (node)-[:HAS_SUPPLY_CHAIN_ROLE]-(r:SupplyChainRole)
    OPTIONAL MATCH (node)-[:LOCATED_IN]-(l:Location)
    RETURN
      node.company AS company,
      collect(DISTINCT o.primary_oems) AS oems,
      collect(DISTINCT r.ev_supply_chain_role) AS roles,
      collect(DISTINCT l.location) AS locations,
      score
    """
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        retrieval_query=retrieval_query,
        embedder=build_embedder(settings),
        neo4j_database=settings.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    result = rag.search(
        question, retriever_config={"top_k": top_k}, return_context=True
    )
    item_count = assert_retrieved("vector_cypher", result)
    print_answer(
        "vector_cypher", result.answer, result.retriever_result.metadata, item_count
    )


def run_chat_history(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    retriever = VectorRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=build_embedder(settings),
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )
    history = InMemoryMessageHistory(
        messages=[
            {
                "role": "user",
                "content": "I am researching EV supply chain companies.",
            },
            {
                "role": "assistant",
                "content": "I will focus on Company nodes, OEMs, roles, locations, and categories.",
            },
        ]
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    result = rag.search(
        question,
        message_history=history,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    item_count = assert_retrieved("chat_history", result)
    print_answer(
        "chat_history", result.answer, result.retriever_result.metadata, item_count
    )


def run_text2cypher(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=build_text2cypher_llm(settings),
        neo4j_schema=COMPANY_SCHEMA,
        examples=TEXT2CYPHER_EXAMPLES,
        custom_prompt=TEXT2CYPHER_PROMPT,
        neo4j_database=settings.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    result = rag.search(question, retriever_config={}, return_context=True)
    item_count = assert_retrieved("text2cypher", result)
    print_answer(
        "text2cypher", result.answer, result.retriever_result.metadata, item_count
    )


def run_tool_calling(
    driver: neo4j.Driver, settings: Settings, question: str, top_k: int
) -> None:
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=build_embedder(settings),
        return_properties=["company", "embedding_text"],
        neo4j_database=settings.database,
    )

    def vector_search(query_text: str, top_k: int = top_k) -> Any:
        return vector_retriever.search(query_text=query_text, top_k=top_k)

    vector_tool = Tool(
        name="company_vector_search",
        description="Search Company nodes by semantic similarity for EV supply chain questions.",
        execute_func=vector_search,
        parameters=ObjectParameter(
            description="Company vector search parameters",
            properties={
                "query_text": StringParameter(
                    description="The natural-language company search query.",
                    required=True,
                ),
                "top_k": IntegerParameter(
                    description="Maximum number of companies to retrieve.",
                    minimum=1,
                    maximum=10,
                    required=False,
                ),
            },
            required_properties=["query_text"],
            additional_properties=False,
        ),
    )
    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=build_llm(settings),
        tools=[vector_tool],
        neo4j_database=settings.database,
        system_instruction=(
            "You must call company_vector_search for any question about companies, "
            "OEMs, categories, locations, industries, or EV supply chain roles."
        ),
    )
    result = tools_retriever.search(query_text=question)
    item_count = len(result.items)
    if item_count == 0:
        raise RuntimeError(
            f"No tool result items returned. Metadata: {result.metadata}"
        )
    print_answer(
        "tool_calling",
        f"Tool retrieval returned {item_count} items.",
        result.metadata,
        item_count,
    )


def run_mode(
    mode: str,
    runner: Callable[[neo4j.Driver, Settings, str, int], None],
    driver: neo4j.Driver,
    settings: Settings,
    question: str,
    top_k: int,
) -> bool:
    print(f"\nRunning {mode}...")
    try:
        runner(driver, settings, question, top_k)
        return True
    except Exception as exc:
        print(f"\n[{mode}] FAIL")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Company GraphRAG retrieval modes."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "vector",
            "hybrid",
            "vector-cypher",
            "chat-history",
            "text2cypher",
            "tool-calling",
        ],
        default="all",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    driver = GraphDatabase.driver(
        settings.uri, auth=(settings.username, settings.password)
    )
    modes: dict[str, Callable[[neo4j.Driver, Settings, str, int], None]] = {
        "vector": run_vector,
        "hybrid": run_hybrid,
        "vector-cypher": run_vector_cypher,
        "chat-history": run_chat_history,
        "text2cypher": run_text2cypher,
        "tool-calling": run_tool_calling,
    }

    selected = list(modes) if args.mode == "all" else [args.mode]
    try:
        driver.verify_connectivity()
        results = [
            run_mode(name, modes[name], driver, settings, args.question, args.top_k)
            for name in selected
        ]
    finally:
        driver.close()

    passed = sum(results)
    print(f"\nSummary: {passed}/{len(results)} modes passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
