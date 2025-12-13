from __future__ import annotations

import os

from neo4j import GraphDatabase, basic_auth
import chromadb

# =====================
# Neo4j
# =====================
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "zWF$yls*J;K:DtC3")
# ✅ Banco alvo (Neo4j multi-db)
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "vectoria")

# =====================
# Chroma Cloud (RAG normativo)
# =====================
CHROMA_API_KEY         = os.getenv("CHROMA_API_KEY",         "ck-13V15SvUh23Zc7MXYoio9uoNGHgyJLVNcwJw9ZxYr2Z2")
CHROMA_TENANT          = os.getenv("CHROMA_TENANT",          "c3e00254-1f1b-49fb-8f51-c9fbad3c8d76")
CHROMA_DATABASE        = os.getenv("CHROMA_DATABASE",        "arbopedia")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "arbopedia")

chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)
rag_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD),
    encrypted=False,
    max_connection_lifetime=3600,
)

def neo4j_session(*, access_mode: str = "READ"):
    """Cria uma sessão já apontando para o database correto (multi-db)."""
    return neo4j_driver.session(database=NEO4J_DATABASE, default_access_mode=access_mode)

def close_resources() -> None:
    try:
        neo4j_driver.close()
    except Exception:
        pass
