"""Prepare Company nodes for GraphRAG retrieval with local Ollama embeddings.

This script:
- builds `Company.embedding_text` from the neighboring graph context
- creates/updates `Company.embedding` with a local Ollama embedding model
- creates the vector and full-text indexes used by the retrievers
- optionally runs a small GraphRAG smoke test

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
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import neo4j
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings import OllamaEmbeddings


DEFAULT_URI = "neo4j+s://945619b7.databases.neo4j.io"
DEFAULT_USERNAME = "945619b7"
DEFAULT_DATABASE = "945619b7"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
DEFAULT_GENERATION_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_VECTOR_INDEX = "company_embeddings"
DEFAULT_FULLTEXT_INDEX = "company_fulltext"

FETCH_COMPANIES_QUERY = """
MATCH (c:Company)
WITH c
ORDER BY elementId(c)
{limit_clause}
OPTIONAL MATCH (c)-[:HAS_SUPPLY_CHAIN_ROLE]-(r:SupplyChainRole)
OPTIONAL MATCH (c)-[:BELONGS_TO_INDUSTRY]-(i:IndustryGroup)
OPTIONAL MATCH (c)-[:LOCATED_IN]-(l:Location)
OPTIONAL MATCH (c)-[:SUPPLIES_TO]-(o:OEMGroup)
OPTIONAL MATCH (c)-[:HAS_CATEGORY]-(cat:Category)
RETURN
  elementId(c) AS id,
  c.company AS company,
  c.embedding_text AS existing_text,
  c.embedding_model AS existing_model,
  c.embedding IS NOT NULL AS has_embedding,
  collect(DISTINCT r.ev_supply_chain_role) AS roles,
  collect(DISTINCT i.industry_group) AS industries,
  collect(DISTINCT l.location) AS locations,
  collect(DISTINCT o.primary_oems) AS oems,
  collect(DISTINCT cat.category) AS categories,
  c.product_service_text AS product_service_text,
  c.product_services AS product_services,
  c.employment_total AS employment_total,
  c.ev_battery_relevance AS ev_relevance,
  c.primary_facility_types AS facility_types,
  c.addresses AS addresses,
  c.classification_methods AS classification_methods,
  c.supplier_or_affiliation_types AS supplier_types
"""

UPDATE_COMPANY_QUERY = """
MATCH (c:Company)
WHERE elementId(c) = $id
SET c.embedding_text = $text,
    c.embedding = $embedding,
    c.embedding_model = $embedding_model,
    c.embedding_updated_at = datetime()
"""

SHOW_VECTOR_INDEX_QUERY = """
SHOW VECTOR INDEXES
YIELD name, state, labelsOrTypes, properties, options
WHERE name = $index_name
RETURN
  name,
  state,
  labelsOrTypes,
  properties,
  options.indexConfig.`vector.dimensions` AS dimensions,
  options.indexConfig.`vector.similarity_function` AS similarity
"""

SHOW_FULLTEXT_INDEX_QUERY = """
SHOW INDEXES
YIELD name, type, entityType, labelsOrTypes, properties, state
WHERE name = $index_name AND type = "FULLTEXT"
RETURN name, entityType, labelsOrTypes, properties, state
"""


def cypher_name(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def values_text(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values.strip()
    return ", ".join(str(value) for value in values if value)


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_company_text(row: neo4j.Record | dict[str, Any]) -> str:
    # Include the full row-derived context so vector search can match on
    # products, employment, EV relevance, facility types, etc. -- not just
    # the company name and its graph neighbours.
    lines = [
        ("Company", row.get("company") or ""),
        ("Supply chain roles", values_text(row.get("roles", []))),
        ("Industries", values_text(row.get("industries", []))),
        ("Locations", values_text(row.get("locations", []))),
        ("OEMs supplied to", values_text(row.get("oems", []))),
        ("Categories (tier)", values_text(row.get("categories", []))),
        (
            "Products and services",
            values_text(row.get("product_service_text"))
            or values_text(row.get("product_services", [])),
        ),
        ("Total employment", scalar_text(row.get("employment_total"))),
        ("EV / battery relevance", values_text(row.get("ev_relevance", []))),
        ("Facility types", values_text(row.get("facility_types", []))),
        ("Addresses", values_text(row.get("addresses", []))),
        ("Classification method", values_text(row.get("classification_methods", []))),
        ("Supplier / affiliation type", values_text(row.get("supplier_types", []))),
    ]
    return "\n".join(f"{label}: {value}" for label, value in lines if value).strip()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_indexes(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    dimensions: int,
    vector_index_name: str,
    fulltext_index_name: str,
) -> None:
    vector_index = cypher_name(vector_index_name)
    fulltext_index = cypher_name(fulltext_index_name)

    create_vector_query = f"""
    CREATE VECTOR INDEX {vector_index} IF NOT EXISTS
    FOR (c:Company)
    ON (c.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: $dimensions,
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    driver.execute_query(
        create_vector_query,
        {"dimensions": dimensions},
        database_=database,
    )

    create_fulltext_query = f"""
    CREATE FULLTEXT INDEX {fulltext_index} IF NOT EXISTS
    FOR (c:Company)
    ON EACH [c.company, c.embedding_text]
    """
    driver.execute_query(create_fulltext_query, database_=database)
    driver.execute_query("CALL db.awaitIndexes()", database_=database)


def verify_vector_index(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    index_name: str,
    dimensions: int,
) -> dict[str, Any]:
    result = driver.execute_query(
        SHOW_VECTOR_INDEX_QUERY,
        {"index_name": index_name},
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    if not result.records:
        raise RuntimeError(f"Vector index was not found: {index_name}")

    data = result.records[0].data()
    if data["state"] != "ONLINE":
        raise RuntimeError(f"Vector index {index_name!r} is not ONLINE: {data}")
    if data["dimensions"] != dimensions:
        raise RuntimeError(
            f"Vector index {index_name!r} has dimensions {data['dimensions']}, "
            f"but {dimensions} were expected. Drop/recreate the index or use the "
            "embedding model that created the stored vectors."
        )
    if data["labelsOrTypes"] != ["Company"] or data["properties"] != ["embedding"]:
        raise RuntimeError(
            f"Vector index {index_name!r} targets unexpected data: {data}"
        )
    return data


def verify_fulltext_index(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    index_name: str,
) -> dict[str, Any]:
    result = driver.execute_query(
        SHOW_FULLTEXT_INDEX_QUERY,
        {"index_name": index_name},
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    if not result.records:
        raise RuntimeError(f"Full-text index was not found: {index_name}")

    data = result.records[0].data()
    if data["state"] != "ONLINE":
        raise RuntimeError(f"Full-text index {index_name!r} is not ONLINE: {data}")
    return data


def fetch_company_rows(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    limit: Optional[int],
) -> list[neo4j.Record]:
    limit_clause = "LIMIT $limit" if limit is not None else ""
    query = FETCH_COMPANIES_QUERY.format(limit_clause=limit_clause)
    parameters = {"limit": limit} if limit is not None else {}
    result = driver.execute_query(
        query,
        parameters,
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    return list(result.records)


def update_company_embeddings(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    embedder: OllamaEmbeddings,
    embedding_model: str,
    force: bool,
    limit: Optional[int],
) -> tuple[int, int]:
    rows = fetch_company_rows(driver, database=database, limit=limit)
    updated = 0
    skipped = 0

    for index, row in enumerate(rows, start=1):
        text = build_company_text(row)
        is_current = (
            row.get("has_embedding")
            and row.get("existing_text") == text
            and row.get("existing_model") == embedding_model
        )
        if is_current and not force:
            skipped += 1
            continue

        embedding = embedder.embed_query(text)
        driver.execute_query(
            UPDATE_COMPANY_QUERY,
            {
                "id": row["id"],
                "text": text,
                "embedding": embedding,
                "embedding_model": embedding_model,
            },
            database_=database,
        )
        updated += 1

        if updated == 1 or updated % 25 == 0:
            print(f"Updated {updated} Company embeddings ({index}/{len(rows)} scanned)")

    return updated, skipped


def run_smoke_test(
    driver: neo4j.Driver,
    *,
    database: Optional[str],
    embedding_model: str,
    generation_model: str,
    ollama_host: str,
    vector_index_name: str,
    question: str,
) -> None:
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.llm import OllamaLLM
    from neo4j_graphrag.retrievers import VectorRetriever

    embedder = OllamaEmbeddings(model=embedding_model, host=ollama_host)
    llm = OllamaLLM(
        model_name=generation_model,
        model_params={"options": {"temperature": 0}},
        host=ollama_host,
    )
    retriever = VectorRetriever(
        driver=driver,
        index_name=vector_index_name,
        embedder=embedder,
        return_properties=["company", "embedding_text"],
        neo4j_database=database,
    )
    rag = GraphRAG(retriever=retriever, llm=llm)
    result = rag.search(question, retriever_config={"top_k": 5}, return_context=True)
    print("\nSmoke test answer:\n")
    print(result.answer)
    print("\nRetrieved context items:", len(result.retriever_result.items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Company embeddings and retrieval indexes for GraphRAG."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--question", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uri = os.getenv("NEO4J_URI", DEFAULT_URI)
    username = os.getenv("NEO4J_USERNAME", DEFAULT_USERNAME)
    password = get_required_env("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE) or None
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    generation_model = os.getenv("OLLAMA_GENERATION_MODEL", DEFAULT_GENERATION_MODEL)
    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    vector_index_name = os.getenv("VECTOR_INDEX_NAME", DEFAULT_VECTOR_INDEX)
    fulltext_index_name = os.getenv("FULLTEXT_INDEX_NAME", DEFAULT_FULLTEXT_INDEX)

    print(f"Connecting to {uri} database={database!r}")
    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        driver.verify_connectivity()
        embedder = OllamaEmbeddings(model=embedding_model, host=ollama_host)
        dimensions = len(embedder.embed_query("test"))
        print(f"Embedding model: {embedding_model}")
        print(f"Embedding dimensions: {dimensions}")

        create_indexes(
            driver,
            database=database,
            dimensions=dimensions,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
        )

        updated, skipped = update_company_embeddings(
            driver,
            database=database,
            embedder=embedder,
            embedding_model=embedding_model,
            force=args.force,
            limit=args.limit,
        )

        vector_index = verify_vector_index(
            driver,
            database=database,
            index_name=vector_index_name,
            dimensions=dimensions,
        )
        fulltext_index = verify_fulltext_index(
            driver,
            database=database,
            index_name=fulltext_index_name,
        )

        print("\nDone.")
        print(f"Company embeddings updated: {updated}")
        print(f"Company embeddings skipped: {skipped}")
        print(f"Vector index: {vector_index}")
        print(f"Full-text index: {fulltext_index}")

        if args.question:
            run_smoke_test(
                driver,
                database=database,
                embedding_model=embedding_model,
                generation_model=generation_model,
                ollama_host=ollama_host,
                vector_index_name=vector_index_name,
                question=args.question,
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
